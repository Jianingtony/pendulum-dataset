import os
import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split, Trajectory, _to_x
from rarhmm.train import load_checkpoint as lc
from rarhmm.predict import rollout_posterior

def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi

def wrap_trajectory(tr: Trajectory, cfg: Config) -> Trajectory:
    theta_wrapped = wrap_to_pi(tr.theta)
    x_new = _to_x(theta_wrapped, tr.omega, cfg)
    return Trajectory(
        id=tr.id,
        regime=tr.regime,
        E_bar=tr.E_bar,
        theta=theta_wrapped,
        omega=tr.omega,
        x=x_new,
        split=tr.split
    )

def main():
    run_dir = Path("runs/K7_fixed_b_vi_v3")
    data_root = "data/pendulum_L4/pendulum"
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
    
    ckpt_path = run_dir / "chain.pkl"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found. Train the model first!")
        return
        
    ckpt = lc(ckpt_path)
    cfg = ckpt["cfg"]
    samples = ckpt["samples"]
    
    # Validation trajectories to run rollout on
    traj_ids = ["traj_003050", "traj_003140"]
    
    trajs = load_split(data_root, "val", cfg)
    trajs = [wrap_trajectory(t, cfg) for t in trajs]
    
    prefix = 100
    horizon = 250
    n_samples = 12
    fps = 20
    
    for traj_id in traj_ids:
        tr = next((t for t in trajs if t.id == traj_id), None)
        if tr is None:
            print(f"[WARN] traj {traj_id} not found in val split, skipping")
            continue
            
        T_total = tr.x.shape[0]
        T0 = min(prefix, T_total - 2)
        H = min(horizon, T_total - T0)
        
        rng = np.random.default_rng(0)
        Xs, Zs = rollout_posterior(cfg, samples, tr.x[:T0], H, n_samples, rng)
        
        # For theta_omega representation
        theta_s = wrap_to_pi(Xs[..., 0])
        omega_s = Xs[..., 1]
        
        theta_gt_pre = tr.theta[:T0]
        theta_gt_fut = tr.theta[T0 : T0 + H]
        omega_gt_pre = tr.omega[:T0] / cfg.omega0
        omega_gt_fut = tr.omega[T0 : T0 + H] / cfg.omega0
        t_pre = np.arange(T0) * cfg.dt
        t_fut = np.arange(T0, T0 + H) * cfg.dt
        
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
        ax_p, ax_th, ax_om = axes
        
        Lp = cfg.L
        ax_p.set_xlim(-1.2 * Lp, 1.2 * Lp)
        ax_p.set_ylim(-1.2 * Lp, 1.2 * Lp)
        ax_p.set_aspect("equal")
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_title("Pendulum")
        
        rod_gt, = ax_p.plot([], [], "-", color="k", lw=2.5)
        bob_gt, = ax_p.plot([], [], "o", color="k", ms=12)
        rod_md, = ax_p.plot([], [], "--", color="tab:red", lw=1.5, alpha=0.85)
        bob_md, = ax_p.plot([], [], "o", color="tab:red", ms=9, alpha=0.85)
        ax_p.plot(0, 0, "+", color="grey")
        
        for ax, ylab, gt_pre, gt_fut, sam in [
            (ax_th, r"$\theta$ (rad)", theta_gt_pre, theta_gt_fut, theta_s),
            (ax_om, r"$\omega/\omega_0$", omega_gt_pre, omega_gt_fut, omega_s),
        ]:
            ax.plot(t_pre, gt_pre, color="0.55", lw=1.2, label="prefix (observed)")
            ax.plot(t_fut, gt_fut, color="black", lw=1.4, label="ground truth")
            for d in range(n_samples):
                ax.plot(t_fut, sam[d], color="tab:red", lw=0.6, alpha=0.35)
            ax.set_xlabel("t [s]")
            ax.set_ylabel(ylab)
            ax.axvline(T0 * cfg.dt, color="grey", ls=":", lw=0.8)
            ax.legend(loc="best", fontsize=8)
            
        ax_th.set_title("θ(t): prefix + ground truth + posterior predictive")
        ax_om.set_title("ω(t)/ω0: prefix + ground truth + posterior predictive")
        
        now_lines = [ax_th.axvline(0, color="orange", lw=1.2),
                     ax_om.axvline(0, color="orange", lw=1.2)]
                     
        theta_full_gt = np.concatenate([theta_gt_pre, theta_gt_fut])
        theta_full_md = np.concatenate([np.tile(theta_gt_pre, (n_samples, 1)), theta_s], axis=1)
        sample_pick = 0
        
        def update(frame):
            th_gt = theta_full_gt[frame]
            th_md = theta_full_md[sample_pick, frame]
            x_gt, y_gt = Lp * np.sin(th_gt), -Lp * np.cos(th_gt)
            x_md, y_md = Lp * np.sin(th_md), -Lp * np.cos(th_md)
            
            rod_gt.set_data([0, x_gt], [0, y_gt])
            bob_gt.set_data([x_gt], [y_gt])
            
            if frame >= T0:
                rod_md.set_data([0, x_md], [0, y_md])
                bob_md.set_data([x_md], [y_md])
            else:
                rod_md.set_data([], [])
                bob_md.set_data([], [])
                
            for nl in now_lines:
                nl.set_xdata([frame * cfg.dt, frame * cfg.dt])
                
            ax_p.set_title(f"pendulum (t = {frame*cfg.dt:.2f}s, "
                           f"{'prefix' if frame < T0 else 'forecast'})")
            return rod_gt, bob_gt, rod_md, bob_md, *now_lines
            
        nframes = T0 + H
        ani = anim.FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
        
        out_gif_run = run_dir / f"rollout_{tr.id}_T0={T0}_H={H}.gif"
        out_gif_art = artifacts_dir / f"rollout_{tr.id}_T0={T0}_H={H}.gif"
        
        ani.save(out_gif_run, writer=anim.PillowWriter(fps=fps))
        ani.save(out_gif_art, writer=anim.PillowWriter(fps=fps))
        plt.close(fig)
        
        print(f"Generated and saved rollout GIF for {tr.id} to {out_gif_run} and artifacts.")

if __name__ == "__main__":
    main()
