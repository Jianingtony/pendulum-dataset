import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Constants for L=4.0
    g = 9.8
    L = 4.0
    w0 = math.sqrt(g / L)  # ~1.565
    dt = 0.05
    alpha = 0.08
    
    # Angles in degrees
    theta_mid = 20.0
    theta_high = 90.0
    
    # 7 State centers/anchors in the physical phase space (theta, omega/omega0)
    # Since they are equilibrium or turning points, they all have velocity = 0.0
    # State 6 and 7 are rotation wrap-arounds at +/- 2pi.
    fixed_centers = [
        (0.0, 0.0),                               # State 1 (Oscillation Center)
        (math.radians(theta_mid), 0.0),          # State 2 (Oscillation Mid Right)
        (-math.radians(theta_mid), 0.0),         # State 3 (Oscillation Mid Left)
        (math.radians(theta_high), 0.0),         # State 4 (Oscillation High Right)
        (-math.radians(theta_high), 0.0),        # State 5 (Oscillation High Left)
        (2.0 * np.pi, 0.0),                       # State 6 (Rotation Wrap +2pi)
        (-2.0 * np.pi, 0.0)                       # State 7 (Rotation Wrap -2pi)
    ]
    
    # Biases (for drawing arrows)
    C1 = 0.5 * w0 * dt
    fixed_b = [
        np.array([0.0, 0.0]),
        np.array([C1 * alpha * math.radians(theta_mid), alpha * math.radians(theta_mid)]),
        np.array([-C1 * alpha * math.radians(theta_mid), -alpha * math.radians(theta_mid)]),
        np.array([C1 * alpha * math.radians(theta_high), alpha * math.radians(theta_high)]),
        np.array([-C1 * alpha * math.radians(theta_high), -alpha * math.radians(theta_high)]),
        np.array([2.0 * np.pi, 0.0]),
        np.array([-2.0 * np.pi, 0.0])
    ]
    
    state_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    labels = [
        r"State 1: Center ($0^\circ, 0$)",
        r"State 2: Mid Right ($+20^\circ, 0$)",
        r"State 3: Mid Left ($-20^\circ, 0$)",
        r"State 4: High Right ($+90^\circ, 0$)",
        r"State 5: High Left ($-90^\circ, 0$)",
        r"State 6: Wrap Right ($+360^\circ, 0$)",
        r"State 7: Wrap Left ($-360^\circ, 0$)"
    ]
    
    fig, ax = plt.subplots(figsize=(11, 5.0))
    
    # Plot background energy contours
    th_grid = np.linspace(-2.2 * np.pi, 2.2 * np.pi, 300)
    om_grid = np.linspace(-3.0, 3.0, 150)
    TH, OM = np.meshgrid(th_grid, om_grid)
    # Hamiltonian (Energy)
    E = 0.5 * OM**2 - (g / L) * np.cos(TH)
    ax.contour(TH, OM, E, levels=25, colors='gray', alpha=0.15, linestyles='dashed')
    
    # Draw reference axes lines (y=0 is the horizontal axis)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    
    # Draw vertical grid lines at +/- pi and +/- 2pi
    ax.axvline(np.pi, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label=r"Phase Boundary $\pm \pi$")
    ax.axvline(-np.pi, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # Plot the 7 states
    for k in range(7):
        cx, cy = fixed_centers[k]
        color = state_colors[k]
        ax.scatter(cx, cy, color=color, edgecolor='black', s=120, zorder=5, label=labels[k])
        
        # Plot bias arrows for oscillation states (scaled for visibility)
        b = fixed_b[k]
        if k < 5 and k > 0:
            scale_arrow = 20.0
            ax.quiver(cx, cy, b[0] * scale_arrow, b[1] * scale_arrow, 
                      angles='xy', scale_units='xy', scale=1, color=color, 
                      width=0.004, alpha=0.8, zorder=6)
        elif k >= 5:
            # For wrap-around states, draw dashed arrow pointing back towards origin representing the shift
            direction = -1.0 if k == 5 else 1.0
            ax.annotate("", xy=(cx + direction * 1.5, cy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="--", alpha=0.8))
            ax.text(cx + direction * 0.75, cy + 0.15, r"$-2\pi$ shift" if k == 5 else r"$+2\pi$ shift", 
                    color=color, fontsize=8, ha='center', fontweight='bold')
            
    ax.set_xlim(-2.3 * np.pi, 2.3 * np.pi)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xticks([-2.0*np.pi, -np.pi, 0.0, np.pi, 2.0*np.pi])
    ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$'])
    
    ax.set_xlabel(r"Pendulum Angle $\theta$ (rad)", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"Normalized Angular Velocity $\omega / \omega_0$", fontsize=11, fontweight='bold')
    ax.set_title("K7 Fixed-Bias State Anchors in Physical Phase Space (All 7 on Horizontal Axis)", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
    
    # Save directory
    out_dir = Path("runs/K7_fixed_b_vi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase_space_anchors.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    
    # Copy to artifacts
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
    plt.savefig(artifacts_dir / "K7_phase_space_anchors.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved phase space plot to {out_path} and artifacts.")

if __name__ == '__main__':
    main()
