import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

# Constants
g, L, dt = 9.8, 1.0, 0.05
omega0 = np.sqrt(g / L) # ~3.13 rad/s

# --- K10 Empirical Centers and Biases (from inspect_k10_centers.py) ---
# Format: {state_id: (name, theta_center, omega_center, b_theta, b_omega, color)}
k10_data = {
    10: ("State 10: Equilibrium", 0.0, 0.0, -8.27788108e-06, -1.05020304e-04, "#1f77b4"),
    9: ("State 9: Med Left", -0.3298, 0.0, -0.00251535, -0.03237535, "#ff7f0e"),
    8: ("State 8: Med Right", 0.5912, 0.0, 0.00376262, 0.04778434, "#2ca02c"),
    6: ("State 6: High Left", -1.55, 0.0, 0.01511768, 0.19323118, "#d62728"),
    7: ("State 7: High Right", 2.04, 0.0, -0.00895674, -0.11438156, "#9467bd"),
    3: ("State 3: Top Right", np.pi, 0.0, -0.03366241, -0.43133376, "#8c564b"),
    5: ("State 5: Top Left", -np.pi, 0.0, 0.03248981, 0.41526877, "#e377c2"),
}

# --- K5 Proposed Pure-Angle Centers and Biases ---
# Let's set theta_mid = 30 degrees (0.52 rad) and theta_high = 90 degrees (1.57 rad)
# Using constant C1 = 0.078, alpha = 0.08
theta_mid = np.radians(30)
theta_high = np.radians(90)
alpha = 0.08
C1 = 0.07825

k5_data = {
    1: ("State 1: Equilibrium", 0.0, 0.0, 0.0, 0.0, "#1f77b4"),
    2: ("State 2: Med Right (+30°)", theta_mid, 0.0, C1 * alpha * theta_mid, alpha * theta_mid, "#2ca02c"),
    3: ("State 3: Med Left (-30°)", -theta_mid, 0.0, -C1 * alpha * theta_mid, -alpha * theta_mid, "#ff7f0e"),
    4: ("State 4: High Right (+90°)", theta_high, 0.0, C1 * alpha * theta_high, alpha * theta_high, "#9467bd"),
    5: ("State 5: High Left (-90°)", -theta_high, 0.0, -C1 * alpha * theta_high, -alpha * theta_high, "#d62728"),
}

# Create figures side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), sharex=True, sharey=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Background energy contours helper
thetas = np.linspace(-np.pi * 1.1, np.pi * 1.1, 200)
omegas = np.linspace(-5.0, 5.0, 200)
TH, OM = np.meshgrid(thetas, omegas)
E = 0.5 * OM**2 - (g / L) * np.cos(TH)

# Draw on both subplots
for ax in (ax1, ax2):
    ax.contour(TH, OM, E, levels=20, colors='gray', alpha=0.15, linestyles='dashed', linewidths=0.8)
    ax.contour(TH, OM, E, levels=[g/L], colors='black', alpha=0.25, linestyles='solid', linewidths=1.0)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlabel("角度 $\\theta$ (rad)", fontsize=12)
    ax.set_xlim(-np.pi * 1.1, np.pi * 1.1)
    ax.set_ylim(-4.0, 4.0)

ax1.set_ylabel("角速度 $\\omega$ (rad/s)", fontsize=12)

# --- Plot K10 Empirical ---
ax1.set_title("Empirical K10 States (Learned in K10_theta_allE_wrap_vi)", fontsize=13, pad=10)
arrow_scale_k10 = 6.0 # Scale to make arrows visible in rad/s plot
for k, (name, theta, omega, b_t, b_w, color) in k10_data.items():
    ax1.plot(theta, omega, 'o', color=color, markersize=10, markeredgecolor='black', zorder=5, label=name)
    # The bias vector has components in (theta, omega/omega0). To plot in (theta, omega), we scale omega component by omega0
    dx = b_t * arrow_scale_k10
    dy = (b_w * omega0) * arrow_scale_k10
    if np.hypot(dx, dy) > 1e-4:
        ax1.quiver(theta, omega, dx, dy, angles='xy', scale_units='xy', scale=1, 
                   color=color, width=0.005, headwidth=4, headlength=5, zorder=6)
ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)

# --- Plot K5 Proposed ---
ax2.set_title("Proposed K5 Pure-Angle States (Parameterized)", fontsize=13, pad=10)
arrow_scale_k5 = 6.0
for k, (name, theta, omega, b_t, b_w, color) in k5_data.items():
    ax2.plot(theta, omega, 'o', color=color, markersize=10, markeredgecolor='black', zorder=5, label=name)
    dx = b_t * arrow_scale_k5
    dy = (b_w * omega0) * arrow_scale_k5
    if np.hypot(dx, dy) > 1e-4:
        ax2.quiver(theta, omega, dx, dy, angles='xy', scale_units='xy', scale=1, 
                   color=color, width=0.005, headwidth=4, headlength=5, zorder=6)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)

plt.suptitle("钟摆相位子空间中状态锚点与偏置 (b_k) 漂移向量对比: K10 (学出) vs K5 (提议)", fontsize=15, y=0.98)
plt.tight_layout()

# Save
out_dir = Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "subspace_comparison.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Comparison plot saved successfully to: {out_path}")
