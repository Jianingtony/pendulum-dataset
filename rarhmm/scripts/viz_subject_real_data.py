import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# Physical constants
g = 9.8
L = 4.0
w0 = math.sqrt(g / L)  # ~1.565

def wrap_pi(val):
    return (val + np.pi) % (2.0 * np.pi) - np.pi

def main():
    subject_npz = "data/subject_trials_preprocessed.npz"
    out_dir = Path("runs/K7_fixed_b_vi_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load subject data
    if not Path(subject_npz).exists():
        print(f"Error: {subject_npz} not found.")
        sys.exit(1)
        
    subj_data = np.load(subject_npz, allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    theta_est = subj_data["theta_estimated"]    # (N,)
    theta_act = subj_data["theta_actual"]       # (N,)
    energy_phys = subj_data["energy_phys"]      # (N,)
    N_trials = x_start.shape[0]
    
    print(f"Loaded {N_trials} subject trials.")
    
    # Calculate theta_max for normalized error calculation
    theta_max = []
    for E in energy_phys:
        if E >= 78.4:
            theta_max.append(np.pi)
        else:
            theta_max.append(np.arccos(1.0 - E / 39.2))
    theta_max = np.array(theta_max)
    
    # Compute subject errors vs physics
    raw_errors_deg = np.zeros(N_trials)
    norm_errors_ratio = np.zeros(N_trials)
    for i in range(N_trials):
        raw_err_rad = abs(wrap_pi(theta_est[i] - theta_act[i]))
        raw_errors_deg[i] = np.degrees(raw_err_rad)
        norm_errors_ratio[i] = raw_err_rad / theta_max[i]
        
    mae_deg = raw_errors_deg.mean()
    rss_deg = (raw_errors_deg ** 2).sum()
    mae_ratio = norm_errors_ratio.mean()
    
    # Setup plotting grid
    th_grid = np.linspace(-np.pi, np.pi, 200)
    om_grid = np.linspace(-4.5, 4.5, 200)
    TH_c, OM_c = np.meshgrid(th_grid, om_grid)
    E_c = 0.5 * OM_c**2 - (g / L) * np.cos(TH_c)
    
    # Start coordinates (terminal point of prefix, i.e., blackout start)
    x_terminal = x_start[:, 1]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("hot_r")
    
    # Panel A: Subject Raw Estimation Error vs Physics
    ax = axes[0]
    vmax = math.ceil(np.percentile(raw_errors_deg, 95) / 5) * 5
    norm_a = mcolors.Normalize(vmin=0.0, vmax=vmax)
    order_a = np.argsort(raw_errors_deg)
    sc1 = ax.scatter(x_terminal[order_a, 0], x_terminal[order_a, 1], c=raw_errors_deg[order_a], 
                     cmap=cmap, norm=norm_a, s=20, alpha=0.8, edgecolors='black', linewidths=0.3, zorder=3)
    ax.contour(TH_c, OM_c, E_c, levels=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], colors="grey", linewidths=0.4, alpha=0.4)
    ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel(r"Angle $\theta_0$ (rad)", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"Normalized Velocity $\omega_0 / \omega_0$", fontsize=10, fontweight='bold')
    ax.set_title(f"Panel A: Subject Raw Error vs Physics (deg)\nMAE = {mae_deg:.2f}°, RSS = {rss_deg:.2f}", fontsize=11, fontweight='bold', pad=10)
    cbar1 = fig.colorbar(sc1, ax=ax)
    cbar1.set_label("Absolute Error (degrees)")
    ax.grid(True, alpha=0.3)
    
    # Panel B: Subject Normalized Estimation Error vs Physics
    ax = axes[1]
    norm_b = mcolors.Normalize(vmin=0.0, vmax=0.8)  # 0% to 80% scale
    order_b = np.argsort(norm_errors_ratio)
    sc2 = ax.scatter(x_terminal[order_b, 0], x_terminal[order_b, 1], c=norm_errors_ratio[order_b], 
                     cmap=cmap, norm=norm_b, s=20, alpha=0.8, edgecolors='black', linewidths=0.3, zorder=3)
    ax.contour(TH_c, OM_c, E_c, levels=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], colors="grey", linewidths=0.4, alpha=0.4)
    ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel(r"Angle $\theta_0$ (rad)", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"Normalized Velocity $\omega_0 / \omega_0$", fontsize=10, fontweight='bold')
    ax.set_title(f"Panel B: Subject Normalized Error (% of max angle)\nMAE = {mae_ratio:.2%}", fontsize=11, fontweight='bold', pad=10)
    cbar2 = fig.colorbar(sc2, ax=ax, format=PercentFormatter(1.0))
    cbar2.set_label("Error as % of Max Angle")
    ax.grid(True, alpha=0.3)
    
    # Panel C: Subject Estimations vs Actual Terminal Angles
    ax = axes[2]
    # Wrap actual and estimated angles to [-pi, pi] for plotting
    theta_act_wrapped = wrap_pi(theta_act)
    theta_est_wrapped = wrap_pi(theta_est)
    
    # Plot scatter of estimated vs actual
    ax.scatter(theta_act_wrapped, theta_est_wrapped, color='royalblue', s=25, alpha=0.6, edgecolors='black', linewidths=0.3, zorder=3)
    # Draw identity line (perfect estimate)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', linewidth=1.5, label='Perfect Estimate (Physics)', zorder=4)
    
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_xlabel(r"Actual Terminal Angle $\theta_{actual}$ (rad)", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"Subject Estimated Angle $\theta_{estimated}$ (rad)", fontsize=10, fontweight='bold')
    ax.set_title("Panel C: Subject Estimates vs Actual Angles", fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    
    fig.suptitle("Human Subject Real Data Performance & Errors (Physics baseline)", fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout()
    
    # Save the figure
    fig.savefig(out_dir / "subject_real_data_viz_results.png", dpi=200)
    # Also save to artifacts
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
    fig.savefig(artifacts_dir / "subject_real_data_viz_results.png", dpi=200)
    plt.close(fig)
    
    print(f"Subject visualization saved at {out_dir / 'subject_real_data_viz_results.png'}")
    print(f"Subject visualization also saved to artifacts as subject_real_data_viz_results.png")

if __name__ == "__main__":
    main()
