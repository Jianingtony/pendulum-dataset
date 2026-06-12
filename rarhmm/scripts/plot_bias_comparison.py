import sys
import math
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Insert parent dir to import rarhmm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split
from rarhmm.train_vi import _forward_backward_ro
from rarhmm.stick_breaking import stick_breaking_probs
from rarhmm.inference import _per_traj_logobs_logtrans, ffbs_single
from scripts.train_fixed_bias_vi_k5 import build_fixed_b

def get_learned_bias_and_centers():
    # Load K5_theta_allE_vi checkpoint
    ckpt_path = Path("rarhmm/runs/K5_theta_allE_vi/chain.pkl")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    
    cfg: Config = ckpt["cfg"]
    samples = ckpt["samples"]
    
    # Compute posterior mean parameters
    A = np.mean([s.A for s in samples], axis=0) # (K, M, M+1)
    Q = np.mean([s.Q for s in samples], axis=0)
    R = np.mean([s.R for s in samples], axis=0)
    r = np.mean([s.r for s in samples], axis=0)
    
    K, M = cfg.K, cfg.obs_dim
    p_mean = samples[-1]
    p_mean.A = A; p_mean.Q = Q; p_mean.R = R; p_mean.r = r
    
    # Load training data to compute empirical centers
    trajs = load_split("data/pendulum", "train", cfg)[:50]
    
    log_init = ckpt.get("log_init", np.full(K, -np.log(K)))
    rng = np.random.default_rng(42)
    
    all_theta = []
    all_omega = []
    all_z = []
    
    for tr in trajs:
        T = tr.x.shape[0]
        if T <= cfg.ar_lag:
            continue
        bundle = _per_traj_logobs_logtrans(tr, p_mean, cfg)
        if bundle is None:
            continue
        log_obs, log_trans, _ = bundle
        z_hmm = ffbs_single(log_init, log_trans, log_obs, rng)
        z_full = np.empty(T, dtype=np.int64)
        z_full[:cfg.ar_lag - 1] = z_hmm[0]
        z_full[cfg.ar_lag - 1:] = z_hmm
        
        for t in range(T):
            all_theta.append(tr.theta[t])
            all_omega.append(tr.omega[t])
            all_z.append(z_full[t])
            
    all_theta = np.array(all_theta)
    all_omega = np.array(all_omega)
    all_z = np.array(all_z)
    
    # Compute center for each state
    centers = []
    biases = []
    for k in range(K):
        mask = all_z == k
        if mask.sum() > 0:
            centers.append((all_theta[mask].mean(), all_omega[mask].mean()))
        else:
            centers.append((0.0, 0.0))
        # Bias b_k is the last column of A_k
        biases.append(A[k, :, -1])
        
    return centers, biases, cfg.omega0

def main():
    print("Extracting learned biases and empirical centers from K5_theta_allE_vi...")
    learned_centers, learned_biases, omega0_learned = get_learned_bias_and_centers()
    
    # 2. Build fixed bias b for L=4.0
    # mid=30, high=90
    theta_mid_deg = 30.0
    theta_high_deg = 90.0
    g, L_fixed = 9.8, 4.0
    omega0_fixed = math.sqrt(g / L_fixed) # ~1.565
    dt = 0.05
    fixed_biases = build_fixed_b(theta_mid_deg, theta_high_deg, omega0_fixed, dt=dt)
    
    fixed_centers = [
        (0.0, 0.0), # State 1
        (math.radians(theta_mid_deg), 0.0), # State 2
        (-math.radians(theta_mid_deg), 0.0), # State 3
        (math.radians(theta_high_deg), 0.0), # State 4
        (-math.radians(theta_high_deg), 0.0) # State 5
    ]
    
    # Colors matching standard state colors
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    
    # Draw Left Subplot: Fixed Bias (L=4.0)
    ax_left = axes[0]
    # Background energy contours
    th_grid = np.linspace(-np.pi, np.pi, 200)
    om_grid = np.linspace(-6.0, 6.0, 200)
    TH, OM = np.meshgrid(th_grid, om_grid)
    E_fixed = 0.5 * OM**2 - (g / L_fixed) * np.cos(TH)
    ax_left.contour(TH, OM, E_fixed, levels=20, colors='gray', alpha=0.15, linestyles='dashed')
    ax_left.contour(TH, OM, E_fixed, levels=[g / L_fixed], colors='black', alpha=0.25, linestyles='solid')
    ax_left.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax_left.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    
    # Plot fixed anchors and arrows
    scale_fixed = 15.0 # scale factor to make arrows visible
    for k in range(5):
        cx, cy = fixed_centers[k]
        b = fixed_biases[k] # [b_theta, b_omega]
        color = colors[k]
        ax_left.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='black', zorder=5, label=f"State {k+1}")
        # Draw arrow
        if np.linalg.norm(b) > 0:
            ax_left.quiver(cx, cy, b[0] * scale_fixed, b[1] * scale_fixed, 
                           angles='xy', scale_units='xy', scale=1, color=color, 
                           width=0.006, headwidth=4, headlength=5, zorder=6)
            
    ax_left.set_title(f"固定偏置在相空间中的可视化 (L=4.0)\n(偏置向量箭头放大 {scale_fixed} 倍)", fontsize=12, fontweight='bold')
    ax_left.set_xlabel("角度 $\\theta$ (rad)")
    ax_left.set_ylabel("角速度 $\\omega$ (rad/s)")
    ax_left.set_xlim(-np.pi * 0.7, np.pi * 0.7)
    ax_left.set_ylim(-4.5, 4.5)
    ax_left.grid(True, linestyle=':', alpha=0.4)
    ax_left.legend(loc='upper right')
    
    # Draw Right Subplot: Learned Bias (L=1.0)
    ax_right = axes[1]
    L_learned = 1.0
    E_learned = 0.5 * OM**2 - (g / L_learned) * np.cos(TH)
    ax_right.contour(TH, OM, E_learned, levels=20, colors='gray', alpha=0.15, linestyles='dashed')
    ax_right.contour(TH, OM, E_learned, levels=[g / L_learned], colors='black', alpha=0.25, linestyles='solid')
    ax_right.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax_right.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    
    # Plot learned centers and arrows
    scale_learned = 1.0 # no scaling or 1x because wrap biases are huge (~6.28)
    for k in range(5):
        cx, cy = learned_centers[k]
        b = learned_biases[k]
        color = colors[k]
        ax_right.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='black', zorder=5, label=f"State {k+1}")
        if np.linalg.norm(b) > 0:
            # We use quiver to draw the learned bias vector
            ax_right.quiver(cx, cy, b[0] * scale_learned, b[1] * scale_learned, 
                            angles='xy', scale_units='xy', scale=1, color=color, 
                            width=0.005, headwidth=4, headlength=5, zorder=6)
            
    ax_right.set_title(f"K5_theta_allE_vi 自由学习出的偏置可视化 (L=1.0)\n(偏置向量原比例 1x)", fontsize=12, fontweight='bold')
    ax_right.set_xlabel("角度 $\\theta$ (rad)")
    ax_right.set_ylabel("角速度 $\\omega$ (rad/s)")
    ax_right.set_xlim(-np.pi * 1.1, np.pi * 1.1)
    ax_right.set_ylim(-6.5, 6.5)
    ax_right.grid(True, linestyle=':', alpha=0.4)
    ax_right.legend(loc='upper right')
    
    fig.suptitle("钟摆相空间子空间中固定偏置与自由学习偏置 (b_k) 对比图", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_dir = Path("rarhmm/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subspace_bias_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Bias comparison plot saved to: {out_path}")

if __name__ == "__main__":
    main()
