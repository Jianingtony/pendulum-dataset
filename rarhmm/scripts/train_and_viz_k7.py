import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import sys
import math
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split, Trajectory
from rarhmm.train_vi import fit_vi, _forward_backward_ro
from rarhmm.model import ModelParams, RecurrentARHMM
from rarhmm.stick_breaking import stick_breaking_log_probs
from rarhmm.inference import _per_traj_logobs_logtrans

# Physical constants
g = 9.8
L = 4.0
w0 = math.sqrt(g / L)
dt = 0.05

def wrap_pi(val):
    return (val + np.pi) % (2.0 * np.pi) - np.pi

def build_fixed_b_k7(theta_mid_deg, theta_high_deg, w0, dt=0.05, alpha=0.08):
    """Build the constrained (7, 2) bias matrix b based on mid/high angles and rotation jumps."""
    theta_mid_rad = math.radians(theta_mid_deg)
    theta_high_rad = math.radians(theta_high_deg)
    
    C1 = 0.5 * w0 * dt  # physical integration constant (approx 0.03913 for L=4)
    scale_L = w0 / 3.1305  # scale velocity component physically with L=4 (w0 ~ 1.565 vs 3.1305 for L=1)
    
    # 7 states ordered monotonically by angle/velocity:
    # 0: CW wrap (diff_theta > 5.0)
    # 1: -theta_high
    # 2: -theta_mid
    # 3: 0.0
    # 4: +theta_mid
    # 5: +theta_high
    # 6: CCW wrap (diff_theta < -5.0)
    
    b_cw_wrap   = np.array([2.0 * np.pi, 0.0])
    b_neg_high  = np.array([-C1 * alpha * theta_high_rad, -scale_L * alpha * theta_high_rad])
    b_neg_mid   = np.array([-C1 * alpha * theta_mid_rad, -scale_L * alpha * theta_mid_rad])
    b_eq        = np.array([0.0, 0.0])
    b_pos_mid   = np.array([C1 * alpha * theta_mid_rad, scale_L * alpha * theta_mid_rad])
    b_pos_high  = np.array([C1 * alpha * theta_high_rad, scale_L * alpha * theta_high_rad])
    b_ccw_wrap  = np.array([-2.0 * np.pi, 0.0])
    
    return np.vstack([b_cw_wrap, b_neg_high, b_neg_mid, b_eq, b_pos_mid, b_pos_high, b_ccw_wrap])

def stratified_subset(trajs, target_n: int, seed: int):
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    by_E = defaultdict(list)
    for i, tr in enumerate(trajs):
        by_E[round(tr.E_bar, 6)].append(i)

    picked, pool = [], []
    for E, idxs in by_E.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        picked.append(idxs[0])
        pool.extend(idxs[1:])

    n_bins = len(by_E)
    if target_n < n_bins:
        rng.shuffle(picked)
        picked = picked[:target_n]
    else:
        remainder = target_n - n_bins
        if remainder > 0:
            rng.shuffle(pool)
            picked.extend(pool[:remainder])

    picked.sort(key=lambda i: trajs[i].E_bar)
    return [trajs[i] for i in picked]

def rollout_deterministic_pure(cfg: Config, params: ModelParams, prefix_x: np.ndarray, horizon: int) -> np.ndarray:
    P, M = cfg.ar_lag, cfg.obs_dim
    T0 = prefix_x.shape[0]
    
    tr_prefix = Trajectory(id="prefix", regime="", E_bar=np.nan,
                           theta=np.zeros(T0), omega=np.zeros(T0), x=prefix_x)
    K = params.K
    log_init = np.full(K, -np.log(K))
    bundle = _per_traj_logobs_logtrans(tr_prefix, params, cfg)
    
    if bundle is None:
        z_prev = 0
    else:
        log_obs, log_trans, _ = bundle
        gamma = _forward_backward_ro(log_init, log_obs, log_trans)
        z_prev = int(np.argmax(gamma[-1]))
        
    x_hist = list(prefix_x[-P:])
    x_pred = np.empty((horizon, M))
    
    for h in range(horizon):
        x_now = x_hist[-1]
        nu = params.recurrence_logits(x_now[None, :], np.array([z_prev]))[0]
        log_pi = stick_breaking_log_probs(nu)
        log_pi -= log_pi.max()
        pi = np.exp(log_pi)
        pi /= pi.sum()
        
        z_new = int(np.argmax(pi))
        
        lagged = np.concatenate(list(x_hist[-P:]) + [[1.0]])
        x_new = params.A[z_new] @ lagged
        
        x_pred[h] = x_new
        x_hist.append(x_new)
        z_prev = z_new
        
    return x_pred

def main():
    data_root = "data/pendulum_L4/pendulum"
    subject_npz = "data/subject_trials_preprocessed.npz"
    out = "runs/K7_fixed_b_vi_v3"
    
    # Best hyperparams from hypersearch (mid_20.0_high_90.0)
    theta_mid = 20.0
    theta_high = 90.0
    seed = 20260518
    target_n = 100
    
    cfg = Config(
        K=7,
        obs_repr="theta_omega",
        recurrence_mode="ro",
        L=L,
        g=g,
        init_seed=seed,
        out_dir=out,
    )
    
    # 1. Build (7, 2) fixed biases
    fixed_b = build_fixed_b_k7(theta_mid, theta_high, w0, dt=cfg.dt)
    print(f"Built (7, 2) fixed_b matrix:")
    for k in range(7):
        print(f"  State {k+1}: b_k = {fixed_b[k]}")
        
    # 2. Load L=4 training dataset
    trajs_all = load_split(data_root, "train", cfg, max_trajs=None)
    print(f"Loaded {len(trajs_all)} train trajectories.")
    trajs = stratified_subset(trajs_all, target_n, seed=seed)
    print(f"Selected stratified subset of {len(trajs)} trajectories.")
    
    # 3. Fit K7 model
    chain_path = Path(out) / "chain.pkl"
    if chain_path.exists():
        print(f"\nCheckpoint {chain_path} already exists. Skipping training!")
    else:
        print("\n" + "="*60)
        print("  TRAINING K7 FIXED BIAS VI")
        print("="*60)
        fit_vi(
            cfg,
            trajs,
            n_em_iter=100,
            n_r_steps=100,
            r_lr=0.01,
            verbose=True,
            fixed_b=fixed_b
        )
    
    # ---------- Run Visualizations ----------
    print("\n" + "="*60)
    print("  RUNNING VISUALIZATIONS")
    print("="*60)
    
    run_dir_abs = Path(out).resolve()
    
    # 1. viz_dynamics
    print("[viz] Running viz_dynamics...")
    import scripts.viz_dynamics as vd
    sys.argv = ["viz_dynamics", "--run", str(run_dir_abs)]
    try:
        vd.main()
    except Exception as e:
        print(f"[viz] viz_dynamics failed: {e}")
        
    # 2. viz_subspace_error
    print("[viz] Running viz_subspace_error...")
    import scripts.train_and_viz_K5_theta_allE as tvae
    try:
        tvae.run_viz_subspace_error(str(run_dir_abs), "data/pendulum_L4/pendulum")
    except Exception as e:
        print(f"[viz] viz_subspace_error failed: {e}")
        
    # 3. viz_trajectory
    # Pick a few trajectory IDs from validation set
    print("[viz] Running viz_trajectory...")
    val_trajs = load_split("data/pendulum_L4/pendulum", "val", cfg)
    viz_ids = [val_trajs[0].id, val_trajs[1].id, val_trajs[2].id]
    try:
        tvae.run_viz_trajectory(str(run_dir_abs), "data/pendulum_L4/pendulum", viz_ids, split="val")
    except Exception as e:
        print(f"[viz] viz_trajectory failed: {e}")
        
    # 4. viz_rollout_gif
    print("[viz] Running viz_rollout_gif...")
    try:
        tvae.run_viz_rollout_gif(str(run_dir_abs), "data/pendulum_L4/pendulum", viz_ids, split="val", prefix=100, horizon=250)
    except Exception as e:
        print(f"[viz] viz_rollout_gif failed: {e}")
        
    # 5. hypersearch-vi style 3-panel plot
    print("[viz] Generating 3-panel evaluation plot (viz_results.png)...")
    try:
        # Load subject trials
        subj_data = np.load(subject_npz, allow_pickle=True)
        x_start = subj_data["x_start"]              # (N, 2, 2)
        theta_est = subj_data["theta_estimated"]    # (N,)
        theta_act = subj_data["theta_actual"]       # (N,) -- Load physical actual angles
        energy_phys = subj_data["energy_phys"]      # (N,)
        N_trials = x_start.shape[0]
        
        theta_max = []
        for E in energy_phys:
            if E >= 78.4:
                theta_max.append(np.pi)
            else:
                theta_max.append(np.arccos(1.0 - E / 39.2))
        theta_max = np.array(theta_max)
        
        # Load model parameters
        with open(run_dir_abs / "chain.pkl", "rb") as f:
            ckpt = pickle.load(f)
        p_model = ckpt["samples"][-1]
        
        raw_errors_deg = np.zeros(N_trials)
        norm_errors_ratio = np.zeros(N_trials)
        for i in range(N_trials):
            prefix_x = x_start[i]
            x_pred = rollout_deterministic_pure(cfg, p_model, prefix_x, horizon=13)
            theta_pred = x_pred[-1, 0]
            
            # Model's error relative to physics (theta_actual)
            model_err_rad = abs(wrap_pi(theta_pred - theta_act[i]))
            # Subject's error relative to physics
            subj_err_rad = abs(wrap_pi(theta_est[i] - theta_act[i]))
            
            # Absolute difference of these errors
            diff_err_rad = abs(model_err_rad - subj_err_rad)
            raw_errors_deg[i] = np.degrees(diff_err_rad)
            norm_errors_ratio[i] = diff_err_rad / theta_max[i]
            
        mae_deg = raw_errors_deg.mean()
        rss_deg = (raw_errors_deg ** 2).sum()
        mae_ratio = norm_errors_ratio.mean()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # Panel 1: Fixed biases in phase space
        ax = axes[0]
        state_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
        # Draw energy contours
        th_grid = np.linspace(-np.pi, np.pi, 200)
        om_grid = np.linspace(-4.5, 4.5, 200)
        TH_c, OM_c = np.meshgrid(th_grid, om_grid)
        E_c = 0.5 * OM_c**2 - (g / L) * np.cos(TH_c)
        ax.contour(TH_c, OM_c, E_c, levels=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], colors="grey", linewidths=0.4, alpha=0.4)
        
        # Unpermuted fixed biases as the physical lookup reference
        original_fixed_b = build_fixed_b_k7(theta_mid, theta_high, w0, dt=cfg.dt)
        
        # Define physical properties aligned 1-to-1 with the indices of original_fixed_b
        original_centers = [
            (-np.pi, 2.0),                           # Index 0: CW wrap
            (-math.radians(theta_high), 0.0),        # Index 1: Large Left
            (-math.radians(theta_mid), 0.0),         # Index 2: Mid Left
            (0.0, 0.0),                              # Index 3: Center
            (math.radians(theta_mid), 0.0),          # Index 4: Mid Right
            (math.radians(theta_high), 0.0),         # Index 5: Large Right
            (np.pi, -2.0)                            # Index 6: CCW wrap
        ]
        
        original_labels = [
            r"State 6 (Wrap): $b=[+2\pi, 0]$",
            r"State 5 (Osc): $b=[-0.005, -0.126]$",
            r"State 3 (Osc): $b=[-0.001, -0.028]$",
            r"State 1 (Osc): $b=[0, 0]$",
            r"State 2 (Osc): $b=[+0.001, +0.028]$",
            r"State 4 (Osc): $b=[+0.005, +0.126]$",
            r"State 7 (Wrap): $b=[-2\pi, 0]$"
        ]
        
        original_colors = ["#8c564b", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#e377c2"]
        
        scale_arrow = 15.0  # Scale factor to make small physical bias vectors visible
        for k in range(7):
            # Fetch the actual bias of state k from the trained model
            b = p_model.A[k, :, 2]
            
            # Find the closest original bias index
            idx = np.argmin(np.linalg.norm(original_fixed_b - b, axis=1))
            
            cx, cy = original_centers[idx]
            color = original_colors[idx]
            label = original_labels[idx]
            
            ax.scatter(cx, cy, color=color, edgecolor='black', s=80, zorder=5, label=label)
            
            if idx in (1, 2, 4, 5):  # Oscillation states with non-zero biases
                ax.quiver(cx, cy, b[0] * scale_arrow, b[1] * scale_arrow, 
                          angles='xy', scale_units='xy', scale=1, color=color, 
                          width=0.005, alpha=0.8, zorder=6)
            elif idx in (0, 6):      # Wrap-around states
                direction = 1.0 if idx == 0 else -1.0
                ax.annotate("", xy=(cx + direction * 1.2, cy), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2, ls="--", alpha=0.85))
                ax.text(cx + direction * 0.6, cy + 0.15, r"$+2\pi$ jump" if idx == 0 else r"$-2\pi$ jump", 
                        color=color, fontsize=8, ha='center', fontweight='bold')
            
        ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(r"Angle $\theta$ (rad)", fontsize=10, fontweight='bold')
        ax.set_ylabel(r"Normalized Velocity $\omega / \omega_0$", fontsize=10, fontweight='bold')
        ax.set_title("Panel A: Fixed State Anchors & Biases $b_k$", fontsize=11, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Raw error in degrees
        ax = axes[1]
        vmax = math.ceil(np.percentile(raw_errors_deg, 95) / 5) * 5
        cmap = plt.get_cmap("hot_r")
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        
        # Evaluate states of starting points
        z_start = []
        for i in range(N_trials):
            prefix_x = x_start[i]
            tr_prefix = Trajectory(id="prefix", regime="", E_bar=np.nan, theta=np.zeros(2), omega=np.zeros(2), x=prefix_x)
            bundle = _per_traj_logobs_logtrans(tr_prefix, p_model, cfg)
            if bundle is None:
                z_start.append(0)
            else:
                log_obs, log_trans, _ = bundle
                gamma = _forward_backward_ro(np.full(7, -np.log(7)), log_obs, log_trans)
                z_start.append(np.argmax(gamma[-1]))
        z_start = np.array(z_start)
        
        # Plot subject trials in subspace colored by error
        # x_start is shape (N, 2, 2) -> we take the terminal point of the prefix: x_start[i, 1]
        x_terminal = x_start[:, 1]
        order = np.argsort(raw_errors_deg)
        sc1 = ax.scatter(x_terminal[order, 0], x_terminal[order, 1], c=raw_errors_deg[order], cmap=cmap, norm=norm, s=15, alpha=0.75, edgecolors='none', zorder=3)
        ax.contour(TH_c, OM_c, E_c, levels=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], colors="grey", linewidths=0.4, alpha=0.4)
        ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(r"$\theta$ (rad)", fontsize=10, fontweight='bold')
        ax.set_ylabel(r"$\omega / \omega_0$", fontsize=10, fontweight='bold')
        ax.set_title(f"Panel B: Terminal Error Diff (deg)\nMAE = {mae_deg:.2f}°, RSS = {rss_deg:.2f}", fontsize=11, fontweight='bold', pad=10)
        cbar1 = fig.colorbar(sc1, ax=ax)
        cbar1.set_label("Absolute Error Difference (degrees)")
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Normalized error difference as % of max angle
        ax = axes[2]
        norm_r = mcolors.Normalize(vmin=0.0, vmax=0.8)  # 0% to 80% scale
        sc2 = ax.scatter(x_terminal[order, 0], x_terminal[order, 1], c=norm_errors_ratio[order], cmap=cmap, norm=norm_r, s=15, alpha=0.75, edgecolors='none', zorder=3)
        ax.contour(TH_c, OM_c, E_c, levels=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], colors="grey", linewidths=0.4, alpha=0.4)
        ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(r"$\theta$ (rad)", fontsize=10, fontweight='bold')
        ax.set_ylabel(r"$\omega / \omega_0$", fontsize=10, fontweight='bold')
        ax.set_title(f"Panel C: Normalized Error Diff (% of max angle)\nMAE = {mae_ratio:.2%}", fontsize=11, fontweight='bold', pad=10)
        cbar2 = fig.colorbar(sc2, ax=ax, format=PercentFormatter(1.0))
        cbar2.set_label("Error Difference as % of Max Angle")
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"rAR-HMM K=7 Fixed-Bias Evaluation (theta_mid={theta_mid}°, theta_high={theta_high}° + Rotation Wrap-arounds)", fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout()
        
        # Save to out dir
        fig.savefig(run_dir_abs / "viz_results.png", dpi=200)
        # Also copy to artifacts dir
        artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
        fig.savefig(artifacts_dir / "K7_fixed_b_v3_viz_results.png", dpi=200)
        plt.close(fig)
        print(f"[viz] 3-panel plot generated at {run_dir_abs / 'viz_results.png'}")
        print(f"[viz] 3-panel plot also saved to artifacts as K7_fixed_b_v3_viz_results.png")
        
    except Exception as e:
        import traceback
        print(f"[viz] 3-panel plot generation failed: {e}")
        traceback.print_exc()
        
    # Run the new concept and large-angle rollout visualizations
    print("[viz] Running plot_fixed_biases_concept...")
    try:
        import scripts.plot_fixed_biases_concept as pfbc
        pfbc.main()
    except Exception as e:
        print(f"[viz] plot_fixed_biases_concept failed: {e}")
        
    print("[viz] Running viz_rollout_gifs_large_angle...")
    try:
        import scripts.viz_rollout_gifs_large_angle as vrgla
        vrgla.main()
    except Exception as e:
        print(f"[viz] viz_rollout_gifs_large_angle failed: {e}")

    print("[viz] Running plot_error_vs_iterations...")
    try:
        import scripts.plot_error_vs_iterations as pevi
        pevi.main()
    except Exception as e:
        print(f"[viz] plot_error_vs_iterations failed: {e}")

    print("\n" + "="*60)
    print("  ALL COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
