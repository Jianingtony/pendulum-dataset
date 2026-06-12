import os
import sys
import math
import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

# Insert parent dir to import rarhmm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import Trajectory
from rarhmm.model import ModelParams
from rarhmm.inference import _per_traj_logobs_logtrans
from rarhmm.train_vi import _forward_backward_ro
from rarhmm.stick_breaking import stick_breaking_log_probs
from scripts.train_fixed_bias_vi_k5 import build_fixed_b

# Constants
g = 9.8
L = 4.0
w0 = math.sqrt(g / L)  # ~1.565

def wrap_pi(val):
    return (val + np.pi) % (2.0 * np.pi) - np.pi

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
    data_root = Path("data")
    subject_npz_path = data_root / "subject_trials_preprocessed.npz"
    hypersearch_dir = Path("runs/hypersearch_vi")
    
    if not subject_npz_path.exists():
        print(f"Error: {subject_npz_path} does not exist.")
        sys.exit(1)
        
    # Load preprocessed subject trials
    subj_data = np.load(subject_npz_path, allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    theta_est = subj_data["theta_estimated"]    # (N,)
    energy_phys = subj_data["energy_phys"]      # (N,)
    N_trials = x_start.shape[0]
    print(f"Loaded {N_trials} aligned subject trials.")
    
    # Calculate theta_max for each trial based on its physics energy
    theta_max = []
    for E in energy_phys:
        if E >= 78.4:
            theta_max.append(np.pi)
        else:
            theta_max.append(np.arccos(1.0 - E / 39.2))
    theta_max = np.array(theta_max)
    
    # Locate all hypersearch subdirectories
    subdirs = sorted([d for d in hypersearch_dir.iterdir() if d.is_dir() and d.name.startswith("mid_")])
    
    if len(subdirs) == 0:
        print(f"Error: No subdirectories found in {hypersearch_dir}")
        sys.exit(1)
        
    print(f"Found {len(subdirs)} hypersearch directories to visualize.")
    
    # State colors matching standard plot
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    
    # Precompute grid for energy contours
    th_grid = np.linspace(-np.pi, np.pi, 200)
    om_grid = np.linspace(-4.5, 4.5, 200)
    TH, OM = np.meshgrid(th_grid, om_grid)
    E_contours = 0.5 * OM**2 - (g / L) * np.cos(TH)
    
    for subdir in subdirs:
        # Parse mid and high angles from name
        parts = subdir.name.split("_")
        theta_mid = float(parts[1])
        theta_high = float(parts[3])
        
        chain_path = subdir / "chain.pkl"
        if not chain_path.exists():
            print(f"Skipping {subdir.name}: chain.pkl not found.")
            continue
            
        print(f"Processing {subdir.name} (mid={theta_mid} deg, high={theta_high} deg)...")
        with open(chain_path, "rb") as f:
            ckpt = pickle.load(f)
            
        cfg = ckpt["cfg"]
        samples = ckpt["samples"]
        p_model = samples[-1]  # converged parameters
        
        # 1. Run rollouts and compute both raw and normalized errors
        raw_errors_deg = np.zeros(N_trials)
        norm_errors_ratio = np.zeros(N_trials)
        
        for i in range(N_trials):
            prefix_x = x_start[i]
            x_pred = rollout_deterministic_pure(cfg, p_model, prefix_x, horizon=13)
            theta_pred = x_pred[-1, 0]
            
            raw_err_rad = abs(wrap_pi(theta_pred - theta_est[i]))
            raw_errors_deg[i] = np.degrees(raw_err_rad)
            norm_errors_ratio[i] = raw_err_rad / theta_max[i]
            
        mae_deg = raw_errors_deg.mean()
        rss_deg = (raw_errors_deg ** 2).sum()
        
        mae_ratio = norm_errors_ratio.mean()
        rss_ratio = (norm_errors_ratio ** 2).sum()
        
        # 2. Build fixed bias b used for training
        fixed_b = build_fixed_b(theta_mid, theta_high, w0, dt=cfg.dt)
        fixed_centers = [
            (0.0, 0.0),
            (math.radians(theta_mid), 0.0),
            (-math.radians(theta_mid), 0.0),
            (math.radians(theta_high), 0.0),
            (-math.radians(theta_high), 0.0)
        ]
        
        # 3. Plotting: 3-panel layout
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        
        # Left Panel: Fixed Bias Vectors
        ax_left = axes[0]
        ax_left.contour(TH, OM, E_contours, levels=15, colors='gray', alpha=0.15, linestyles='dashed')
        ax_left.contour(TH, OM, E_contours, levels=[g / L], colors='black', alpha=0.25, linestyles='solid')
        ax_left.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax_left.axvline(0, color='black', linewidth=0.8, alpha=0.4)
        
        scale_arrow = 15.0  # Scale factor to make physical bias vectors visible
        for k in range(5):
            cx, cy = fixed_centers[k]
            b = fixed_b[k]
            color = colors[k]
            ax_left.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='black', zorder=5, label=f"State {k+1}")
            if np.linalg.norm(b) > 0:
                ax_left.quiver(cx, cy, b[0] * scale_arrow, b[1] * scale_arrow, 
                               angles='xy', scale_units='xy', scale=1, color=color, 
                               width=0.006, headwidth=4, headlength=5, zorder=6)
        
        ax_left.set_title(f"固定偏置在相空间中 (L=4.0)\n(偏置向量箭头放大 {scale_arrow} 倍)", fontsize=11, fontweight='bold')
        ax_left.set_xlabel("角度 $\\theta$ (rad)")
        ax_left.set_ylabel("角速度 $\\omega$ (rad/s)")
        ax_left.set_xlim(-np.pi * 0.7, np.pi * 0.7)
        ax_left.set_ylim(-3.5, 3.5)
        ax_left.grid(True, linestyle=':', alpha=0.4)
        ax_left.legend(loc='upper right')
        
        # Get starting positions of trials for scatter plots
        th0 = x_start[:, -1, 0]
        om0 = x_start[:, -1, 1]
        
        # Middle Panel: Raw Angle Error in Degrees (No floor/ceiling effect)
        ax_mid = axes[1]
        ax_mid.contour(TH, OM, E_contours, levels=15, colors='gray', alpha=0.15, linestyles='dashed')
        ax_mid.contour(TH, OM, E_contours, levels=[g / L], colors='black', alpha=0.25, linestyles='solid')
        ax_mid.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax_mid.axvline(0, color='black', linewidth=0.8, alpha=0.4)
        
        # Determine dynamic vmax for raw errors to avoid floor/ceiling effects
        vmax_deg = math.ceil(np.percentile(raw_errors_deg, 95) / 5.0) * 5.0
        if vmax_deg < 5.0:
            vmax_deg = 5.0
        
        norm_mid = mcolors.Normalize(vmin=0.0, vmax=vmax_deg)
        cmap = plt.get_cmap("coolwarm")
        
        order_mid = np.argsort(raw_errors_deg)
        sc_mid = ax_mid.scatter(th0[order_mid], om0[order_mid], c=raw_errors_deg[order_mid],
                                cmap=cmap, norm=norm_mid, s=15, alpha=0.8,
                                edgecolors="black", linewidths=0.5, zorder=5)
        
        for k in range(5):
            cx, cy = fixed_centers[k]
            ax_mid.plot(cx, cy, 'o', color='none', markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=6)
            ax_mid.text(cx, cy + 0.15, f"{k+1}", fontsize=10, fontweight='bold', ha='center', va='bottom', zorder=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
            
        ax_mid.set_title(f"被试测试集 13步终端角度误差 (角度值)\nMAE = {mae_deg:.2f}°, RSS = {rss_deg:.2f}°²", fontsize=11, fontweight='bold')
        ax_mid.set_xlabel("初始角度 $\\theta_0$ (rad)")
        ax_mid.set_ylabel("初始角速度 $\\omega_0$ (rad/s)")
        ax_mid.set_xlim(-np.pi * 0.7, np.pi * 0.7)
        ax_mid.set_ylim(-3.5, 3.5)
        ax_mid.grid(True, linestyle=':', alpha=0.4)
        
        cbar_mid = fig.colorbar(sc_mid, ax=ax_mid, shrink=0.85, pad=0.02)
        cbar_mid.set_label("角度绝对误差 $| \\theta_{pred} - \\theta_{est} |$ (度)", fontsize=10)
        
        # Right Panel: Normalized Error Ratio (Percentage)
        ax_right = axes[2]
        ax_right.contour(TH, OM, E_contours, levels=15, colors='gray', alpha=0.15, linestyles='dashed')
        ax_right.contour(TH, OM, E_contours, levels=[g / L], colors='black', alpha=0.25, linestyles='solid')
        ax_right.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax_right.axvline(0, color='black', linewidth=0.8, alpha=0.4)
        
        norm_right = mcolors.Normalize(vmin=0.0, vmax=0.8)
        
        order_right = np.argsort(norm_errors_ratio)
        sc_right = ax_right.scatter(th0[order_right], om0[order_right], c=norm_errors_ratio[order_right],
                                    cmap=cmap, norm=norm_right, s=15, alpha=0.8,
                                    edgecolors="black", linewidths=0.5, zorder=5)
        
        for k in range(5):
            cx, cy = fixed_centers[k]
            ax_right.plot(cx, cy, 'o', color='none', markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=6)
            ax_right.text(cx, cy + 0.15, f"{k+1}", fontsize=10, fontweight='bold', ha='center', va='bottom', zorder=7,
                          bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
            
        ax_right.set_title(f"被试测试集 13步终端角度误差 (占最大摆角比)\nMAE = {mae_ratio:.2%}, RSS = {rss_ratio:.4f}", fontsize=11, fontweight='bold')
        ax_right.set_xlabel("初始角度 $\\theta_0$ (rad)")
        ax_right.set_ylabel("初始角速度 $\\omega_0$ (rad/s)")
        ax_right.set_xlim(-np.pi * 0.7, np.pi * 0.7)
        ax_right.set_ylim(-3.5, 3.5)
        ax_right.grid(True, linestyle=':', alpha=0.4)
        
        cbar_right = fig.colorbar(sc_right, ax=ax_right, shrink=0.85, pad=0.02)
        cbar_right.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cbar_right.set_label("归一化终端角度误差 (误差角度 / $\\theta_{max}$)", fontsize=10)
        
        fig.suptitle(f"超参组合 mid={theta_mid}° / high={theta_high}° 的多维度误差与偏置分析", fontsize=14, fontweight='bold', y=0.97)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        
        out_path = subdir / "viz_results.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Saved plot to: {out_path}")
        print(f"  MAE (Deg): {mae_deg:.2f}°, MAE (Norm): {mae_ratio:.2%}\n")

if __name__ == "__main__":
    main()
