import os
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def main():
    # Physical constants for L=4.0
    g = 9.8
    L = 4.0
    w0 = math.sqrt(g / L)  # ~1.565
    dt = 0.05
    alpha = 0.08
    
    # Biases
    theta_mid_rad = math.radians(20.0)
    theta_high_rad = math.radians(90.0)
    C1 = 0.5 * w0 * dt
    
    b1 = np.array([0.0, 0.0])
    b2 = np.array([C1 * alpha * theta_mid_rad, alpha * theta_mid_rad])
    b3 = np.array([-C1 * alpha * theta_mid_rad, -alpha * theta_mid_rad])
    b4 = np.array([C1 * alpha * theta_high_rad, alpha * theta_high_rad])
    b5 = np.array([-C1 * alpha * theta_high_rad, -alpha * theta_high_rad])
    b6 = np.array([2.0 * np.pi, 0.0])
    b7 = np.array([-2.0 * np.pi, 0.0])
    
    biases = np.vstack([b1, b2, b3, b4, b5, b6, b7])
    
    state_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    labels = [
        "State 1 (Center Osc)",
        "State 2 (Mid Osc CCW)",
        "State 3 (Mid Osc CW)",
        "State 4 (High Osc CCW)",
        "State 5 (High Osc CW)",
        "State 6 (Wrap +2pi)",
        "State 7 (Wrap -2pi)"
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Draw main axes lines
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    
    # Plot all 7 biases on main plot
    for k in range(7):
        ax.scatter(biases[k, 0], biases[k, 1], color=state_colors[k], edgecolor='black', s=120, zorder=5, label=labels[k])
        
    ax.set_xlim(-2.3 * np.pi, 2.3 * np.pi)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel(r"Angle Bias $b[\theta]$ (rad)", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"Velocity Bias $b[\omega]$ (rad/s)", fontsize=11, fontweight='bold')
    ax.set_title(r"K7 Model Fixed Biases $b_k$ in Bias Space (Unwrapped)" + "\n" + r"Showing Rotation Wrap-around Biases at $\pm 2\pi$", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    
    # Add an inset axes zoom-in on the 5 oscillation states near (0, 0)
    # Location: [left, bottom, width, height] as fraction of parent axes
    ax_inset = fig.add_axes([0.3, 0.25, 0.35, 0.35])
    ax_inset.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax_inset.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    
    for k in range(5):
        ax_inset.scatter(biases[k, 0], biases[k, 1], color=state_colors[k], edgecolor='black', s=80, zorder=5)
        # Add labels to the inset points
        offset_y = 0.015 if biases[k, 1] >= 0 else -0.025
        ax_inset.text(biases[k, 0], biases[k, 1] + offset_y, f"S{k+1}", fontsize=8, ha='center', fontweight='bold')
        
    ax_inset.set_xlim(-0.01, 0.01)
    ax_inset.set_ylim(-0.15, 0.15)
    ax_inset.set_title("Zoom-in near Origin (Oscillation States)", fontsize=9, fontweight='bold', color='grey')
    ax_inset.grid(True, linestyle=':', alpha=0.3)
    
    # Draw a box indicating the inset region on main plot
    # and connection lines
    rect_x = [-0.01, 0.01, 0.01, -0.01, -0.01]
    rect_y = [-0.15, -0.15, 0.15, 0.15, -0.15]
    ax.plot(rect_x, rect_y, color='grey', linestyle='--', linewidth=1.0)
    
    # Save directory
    out_dir = Path("runs/K7_fixed_b_vi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixed_biases_unwrapped.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    
    # Copy to artifacts
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
    plt.savefig(artifacts_dir / "K7_fixed_biases_unwrapped.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved bias plot to {out_path} and artifacts.")

if __name__ == '__main__':
    main()
