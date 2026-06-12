import sys
import math
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split, Trajectory
from rarhmm.train_vi import fit_vi, _forward_backward_ro
from rarhmm.model import RecurrentARHMM, ModelParams
from rarhmm.stick_breaking import stick_breaking_log_probs
from rarhmm.inference import _per_traj_logobs_logtrans

from scripts.train_fixed_bias_vi_k5 import build_fixed_b

# Physical constants
g = 9.8
L = 4.0
w0 = math.sqrt(g / L)

def wrap_pi(val):
    return (val + np.pi) % (2.0 * np.pi) - np.pi

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
    """Deterministic rollout: x_next = A·x (no process noise Q) and z_next = argmax(pi) (no FFBS sampling)."""
    P, M = cfg.ar_lag, cfg.obs_dim
    T0 = prefix_x.shape[0]
    
    # Prefix trajectory for state inference
    tr_prefix = Trajectory(id="prefix", regime="", E_bar=np.nan,
                           theta=np.zeros(T0), omega=np.zeros(T0), x=prefix_x)
    K = params.K
    log_init = np.full(K, -np.log(K))
    bundle = _per_traj_logobs_logtrans(tr_prefix, params, cfg)
    
    if bundle is None:
        z_prev = 0
    else:
        log_obs, log_trans, _ = bundle
        # Run forward-backward and take the argmax of the last step's posterior gamma
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
        
        # Argmax instead of sampling
        z_new = int(np.argmax(pi))
        
        # Mean dynamics instead of sampling
        lagged = np.concatenate(list(x_hist[-P:]) + [[1.0]])
        x_new = params.A[z_new] @ lagged
        
        x_pred[h] = x_new
        x_hist.append(x_new)
        z_prev = z_new
        
    return x_pred

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/pendulum_L4/pendulum")
    p.add_argument("--subject-npz", type=str, default="data/subject_trials_preprocessed.npz")
    p.add_argument("--out-dir", type=str, default="runs/hypersearch_vi")
    p.add_argument("--n-em-iter", type=int, default=50, help="Fewer iterations for search efficiency.")
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--target-n", type=int, default=100, help="Subset size for training speed.")
    # Support custom grid parameters via command line
    p.add_argument("--theta-mid-grid", type=float, nargs="+", default=[20.0, 30.0, 40.0])
    p.add_argument("--theta-high-grid", type=float, nargs="+", default=[60.0, 90.0, 120.0])
    args = p.parse_args()
    
    cfg = Config(
        K=5,
        obs_repr="theta_omega",
        recurrence_mode="ro",
        L=4.0,
        g=9.8,
        init_seed=args.seed,
    )
    
    # 1. Load preprocessed subject trials
    subj_data = np.load(args.subject_npz, allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    theta_est = subj_data["theta_estimated"]    # (N,)
    print(f"Loaded {x_start.shape[0]} aligned subject trials for evaluation.")
    
    # 2. Load L=4 training dataset
    trajs_all = load_split(args.data_root, "train", cfg, max_trajs=None)
    print(f"Loaded {len(trajs_all)} train trajectories.")
    if args.target_n is not None and args.target_n < len(trajs_all):
        trajs = stratified_subset(trajs_all, args.target_n, seed=args.seed)
        print(f"Selected a stratified subset of {len(trajs)} trajectories.")
    else:
        trajs = trajs_all
    
    grid_mid = sorted(args.theta_mid_grid)
    grid_high = sorted(args.theta_high_grid)
    
    rss_results = np.zeros((len(grid_mid), len(grid_high)))
    
    best_rss = 1e9
    best_mid = None
    best_high = None
    
    # 3. Grid Search Loop
    for m_idx, theta_mid in enumerate(grid_mid):
        for h_idx, theta_high in enumerate(grid_high):
            print(f"\n==================================================")
            print(f"Evaluating: theta_mid = {theta_mid} deg, theta_high = {theta_high} deg")
            print(f"==================================================")
            
            # Setup fixed biases
            fixed_b = build_fixed_b(theta_mid, theta_high, w0, dt=cfg.dt)
            
            # Setup custom run output directory for this hp combination
            cfg.out_dir = str(Path(args.out_dir) / f"mid_{theta_mid}_high_{theta_high}")
            
            # Fit model with fixed bias
            ckpt = fit_vi(
                cfg,
                trajs,
                n_em_iter=args.n_em_iter,
                n_r_steps=60, # slightly fewer steps to speed up
                r_lr=0.01,
                verbose=False,
                fixed_b=fixed_b
            )
            
            p_model = ckpt["samples"][-1] # final parameters
            
            # Evaluate RSS on subject trials
            total_rss = 0.0
            for i in range(x_start.shape[0]):
                prefix_x = x_start[i] # (2, 2)
                # Rollout 13 steps (0.65 seconds)
                x_pred = rollout_deterministic_pure(cfg, p_model, prefix_x, horizon=13)
                theta_pred = x_pred[-1, 0] # terminal predicted angle
                
                diff = wrap_pi(theta_pred - theta_est[i])
                total_rss += diff ** 2
                
            rss_results[m_idx, h_idx] = total_rss
            print(f"Completed evaluation: RSS = {total_rss:.6f}")
            
            if total_rss < best_rss:
                best_rss = total_rss
                best_mid = theta_mid
                best_high = theta_high
                
    print(f"\n==================================================")
    print(f"Grid Search Complete!")
    print(f"Best RSS: {best_rss:.6f} at theta_mid = {best_mid} deg, theta_high = {best_high} deg")
    print(f"==================================================")
    
    # 4. Save RSS Landscape Heatmap (vibrant coolwarm palette with premium aesthetics)
    plt.figure(figsize=(7, 5.5))
    sns.set_theme(style="whitegrid")
    
    # Pivot matrix for heatmap plotting
    ax = sns.heatmap(
        rss_results,
        annot=True,
        fmt=".3f",
        xticklabels=[f"{h}°" for h in grid_high],
        yticklabels=[f"{m}°" for m in grid_mid],
        cmap="coolwarm",
        cbar_kws={'label': 'Residual Sum of Squares (RSS)'},
        linewidths=1.5,
        linecolor='white',
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("rAR-HMM RSS Error Landscape over $\\theta$ Breakpoints\n(K=5, L=4.0, Variational Inference)", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Large-angle Breakpoint ($\\theta_{high}$)", fontsize=10, fontweight='bold')
    plt.ylabel("Mid-angle Breakpoint ($\\theta_{mid}$)", fontsize=10, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\551b31a1-4808-4167-88df-1f692d3053a2")
    heatmap_out = artifacts_dir / "rss_landscape.png"
    plt.savefig(heatmap_out, dpi=300)
    plt.close()
    print(f"Saved RSS error landscape heatmap to {heatmap_out}")

if __name__ == "__main__":
    main()
