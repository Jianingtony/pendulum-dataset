import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters matching config.py
g, L, dt = 9.8, 1.0, 0.05
omega0 = np.sqrt(g / L) # ~3.13 rad/s

# 5 Anchoring points in phase space (theta in radians, omega in rad/s)
theta_turn = np.pi / 4  # 45 degrees
omega_pass = 2.0        # 2.0 rad/s

anchors = {
    1: {"name": "State 1: Lowest Equilibrium", "theta": 0.0, "omega": 0.0, "color": "#1f77b4"},
    2: {"name": "State 2: Right Turning Point", "theta": theta_turn, "omega": 0.0, "color": "#d62728"},
    3: {"name": "State 3: Left Turning Point", "theta": -theta_turn, "omega": 0.0, "color": "#ff7f0e"},
    4: {"name": "State 4: Bottom Rightward Passage", "theta": 0.0, "omega": omega_pass, "color": "#2ca02c"},
    5: {"name": "State 5: Bottom Leftward Passage", "theta": 0.0, "omega": -omega_pass, "color": "#9467bd"}
}

# Drift vector delta_theta, delta_omega for each state
# State 1: no drift
# State 2: right turn -> gravity pulls left (omega decreases)
# State 3: left turn -> gravity pulls right (omega increases)
# State 4: rightward passage -> velocity causes angle to increase
# State 5: leftward passage -> velocity causes angle to decrease
scale_arrow = 5.0 # Scale factor to make arrows visible in the plot

drifts = {
    1: (0.0, 0.0),
    2: (0.0, -(g / L) * np.sin(theta_turn) * dt * scale_arrow),
    3: (0.0, -(g / L) * np.sin(-theta_turn) * dt * scale_arrow),
    4: (omega_pass * dt * scale_arrow, 0.0),
    5: (-omega_pass * dt * scale_arrow, 0.0)
}

plt.figure(figsize=(9, 8))
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # Support Chinese labels
plt.rcParams['axes.unicode_minus'] = False

# Plot constant energy contours in the background
thetas = np.linspace(-np.pi, np.pi, 200)
omegas = np.linspace(-6.0, 6.0, 200)
TH, OM = np.meshgrid(thetas, omegas)
# Hamiltonian (Energy) E = 0.5 * omega^2 - (g/L) * cos(theta)
E = 0.5 * OM**2 - (g / L) * np.cos(TH)

# Draw energy contours
plt.contour(TH, OM, E, levels=25, colors='gray', alpha=0.2, linestyles='dashed', linewidths=0.8)
# Highlight the separatrix (E = g/L = 9.8)
plt.contour(TH, OM, E, levels=[g/L], colors='black', alpha=0.3, linestyles='solid', linewidths=1.2)

# Draw axes
plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)
plt.axvline(0, color='black', linewidth=0.8, alpha=0.5)

# Plot each state
for k, info in anchors.items():
    theta, omega = info["theta"], info["omega"]
    color = info["color"]
    
    # Plot anchor center point
    plt.plot(theta, omega, 'o', color=color, markersize=12, markeredgecolor='black', zorder=5, label=info["name"])
    
    # Plot drift vector (arrow) representing bias b_k
    dx, dy = drifts[k]
    if dx != 0 or dy != 0:
        plt.quiver(theta, omega, dx, dy, angles='xy', scale_units='xy', scale=1, 
                   color=color, width=0.007, headwidth=4, headlength=6, zorder=6)

# Add annotations/diagram of the physical pendulum at the anchors
# Draw small pendulum schematics
def draw_pendulum_inset(ax, theta, omega, x_center, y_center, radius=0.25):
    # ax coordinates
    # Length of string
    l_p = radius * 0.8
    x_bob = x_center + l_p * np.sin(theta)
    y_bob = y_center - l_p * np.cos(theta)
    # Draw pivot
    ax.plot(x_center, y_center, 'k.', markersize=4)
    # Draw string
    ax.plot([x_center, x_bob], [y_center, y_bob], 'k-', linewidth=1.0)
    # Draw bob
    ax.plot(x_bob, y_bob, 'k', marker='o', markersize=5)
    # Draw velocity arrow if any
    if omega != 0:
        v_dir = 1 if omega > 0 else -1
        # perpendicular arrow at the bob
        vx = -v_dir * np.cos(theta) * 0.08
        vy = -v_dir * np.sin(theta) * 0.08
        ax.quiver(x_bob, y_bob, vx, vy, angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, headwidth=3)

ax = plt.gca()
# State 1 inset
draw_pendulum_inset(ax, 0.0, 0.0, 0.0, -0.6, 0.4)
# State 2 inset (Right Turn)
draw_pendulum_inset(ax, theta_turn, 0.0, theta_turn + 0.3, 0.5, 0.4)
# State 3 inset (Left Turn)
draw_pendulum_inset(ax, -theta_turn, 0.0, -theta_turn - 0.3, 0.5, 0.4)
# State 4 inset (Right passage)
draw_pendulum_inset(ax, 0.0, omega_pass, 0.5, omega_pass + 0.6, 0.4)
# State 5 inset (Left passage)
draw_pendulum_inset(ax, 0.0, -omega_pass, -0.5, -omega_pass - 0.6, 0.4)

plt.title("钟摆相位子空间中固定的 5 个状态锚定点及其偏置 (b_k) 运动趋势", fontsize=14, pad=15)
plt.xlabel("角度 $\\theta$ (rad)", fontsize=12)
plt.ylabel("角速度 $\\omega$ (rad/s)", fontsize=12)
plt.xlim(-np.pi * 0.6, np.pi * 0.6)
plt.ylim(-4.5, 4.5)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

# Save figure
out_dir = Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "anchor_subspace.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot saved successfully to: {out_path}")
