import os
import sys
import math
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from scripts.train_and_viz_k7 import build_fixed_b_k7

def main():
    run_dir = Path("runs/K7_fixed_b_vi_v3")
    ckpt_path = run_dir / "chain.pkl"
    
    # Physical constants
    g = 9.8
    L = 4.0
    w0 = math.sqrt(g / L)
    dt = 0.05
    theta_mid = 20.0
    theta_high = 90.0
    
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        cfg = ckpt["cfg"]
        fixed_b = build_fixed_b_k7(theta_mid, theta_high, cfg.omega0, dt=cfg.dt)
        print("Loaded parameters from checkpoint.")
    else:
        fixed_b = build_fixed_b_k7(theta_mid, theta_high, w0, dt=dt)
        print("Using default parameter calculation.")

    # Create a premium 2-panel plot
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.1, 0.9]})
    
    # --- Panel A: Phase Space Anchors & Jumps ---
    # Background energy contours
    th_grid = np.linspace(-np.pi - 0.2, np.pi + 0.2, 200)
    om_grid = np.linspace(-4.5, 4.5, 200)
    TH, OM = np.meshgrid(th_grid, om_grid)
    E = 0.5 * OM**2 - (g / L) * np.cos(TH)
    ax_plot.contour(TH, OM, E, levels=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], colors="grey", linewidths=0.5, alpha=0.3)
    ax_plot.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax_plot.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    
    # Unpermuted fixed biases as the physical lookup reference
    original_fixed_b = build_fixed_b_k7(theta_mid, theta_high, w0, dt=dt)
    
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
        "State 6 (正向越界): CCW $+2\\pi$ 跳转",
        "State 5 (大左偏置): $-90^\\circ$ 摆角",
        "State 3 (中左偏置): $-20^\\circ$ 摆角",
        "State 1 (平衡中心): $0^\\circ$ 摆角",
        "State 2 (中右偏置): $+20^\\circ$ 摆角",
        "State 4 (大右偏置): $+90^\\circ$ 摆角",
        "State 7 (反向越界): CW $-2\\pi$ 跳转"
    ]
    
    original_colors = ["#8c564b", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#e377c2"]
    
    # Draw scatter points & quiver arrows for biases
    scale_arrow = 25.0  # Scale up small bias vectors for visibility
    for k in range(7):
        # Fetch the actual bias of state k from fixed_b (which may be permuted)
        b = fixed_b[k]
        
        # Find the closest original bias index
        idx = np.argmin(np.linalg.norm(original_fixed_b - b, axis=1))
        
        cx, cy = original_centers[idx]
        color = original_colors[idx]
        label = original_labels[idx]
        
        ax_plot.scatter(cx, cy, color=color, edgecolor='black', s=120, zorder=5, label=label)
        
        if idx in (1, 2, 4, 5):  # Oscillation states with non-zero biases
            ax_plot.quiver(cx, cy, b[0] * scale_arrow, b[1] * scale_arrow, 
                           angles='xy', scale_units='xy', scale=1, color=color, 
                           width=0.006, headwidth=4, headlength=5, zorder=6)
        elif idx in (0, 6):      # Wrap-around states
            direction = 1.0 if idx == 0 else -1.0
            ax_plot.annotate("", xy=(cx + direction * 1.5, cy), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.5, ls="--", alpha=0.85))
            ax_plot.text(cx + direction * 0.75, cy + 0.18, "$+2\\pi$ 坐标平移" if idx == 0 else "$-2\\pi$ 坐标平移", 
                        color=color, fontsize=9, ha='center', fontweight='bold')
            
    ax_plot.set_xlim(-np.pi - 0.3, np.pi + 0.3)
    ax_plot.set_ylim(-4.5, 4.5)
    ax_plot.set_xlabel("角度 $\\theta$ (rad)", fontsize=11, fontweight='bold')
    ax_plot.set_ylabel("角速度 $\\omega / \\omega_0$", fontsize=11, fontweight='bold')
    ax_plot.set_title("图 A: 状态锚点在相空间中的位置与偏置矢量 (放大 25 倍)", fontsize=12, fontweight='bold', pad=10)
    ax_plot.legend(fontsize=9, loc='upper right')
    ax_plot.grid(True, linestyle=':', alpha=0.5)
    
    # --- Panel B: Table & Conceptual Explanation ---
    ax_text.axis('off')
    
    # Define text block content
    text_content = (
        "=== 固定的偏置向量 $b_k$ 对应具体数值 ===\n\n"
        f" • 状态 1: b_1 = [{fixed_b[0,0]:.4f}, {fixed_b[0,1]:.4f}] (平衡位置无漂移)\n"
        f" • 状态 2: b_2 = [{fixed_b[1,0]:.4f}, {fixed_b[1,1]:.4f}] (对应摆角 +20°)\n"
        f" • 状态 3: b_3 = [{fixed_b[2,0]:.4f}, {fixed_b[2,1]:.4f}] (对应摆角 -20°)\n"
        f" • 状态 4: b_4 = [{fixed_b[3,0]:.4f}, {fixed_b[3,1]:.4f}] (对应摆角 +90°)\n"
        f" • 状态 5: b_5 = [{fixed_b[4,0]:.4f}, {fixed_b[4,1]:.4f}] (对应摆角 -90°)\n"
        f" • 状态 6: b_6 = [{fixed_b[5,0]:.4f}, {fixed_b[5,1]:.4f}] (CCW 越界跳转量 +2pi)\n"
        f" • 状态 7: b_7 = [{fixed_b[6,0]:.4f}, {fixed_b[6,1]:.4f}] (CW 越界跳转量 -2pi)\n\n"
        "=== 物理概念与通俗解析 ===\n\n"
        "1. 自回归动力学转移方程:\n"
        "   系统下一步的连续物理状态通过自回归方程推演:\n"
        "       x_{t+1} = A_k * x_t + b_k\n"
        "   其中 x_t = [θ_t, ω_t/ω_0]^T 为系统状态向量。\n\n"
        "2. 振荡状态偏置 (States 1-5) 含义:\n"
        "   • 当摆角偏离平衡中心时，重力加速度 (-g/L * sinθ) 将试图拉回单摆。\n"
        "   • 偏置向量 b_k 代表当前状态对应角度下的【重力恢复力漂移量】。由于时间\n"
        "     步长 dt = 0.05s 非常短，因此在单个时间步内产生的物理改变极为微小\n"
        "     (角度修正约 10^-3 rad，角速度修正约 10^-2 rad/s)。\n"
        "   • b_k 作为动力学常量，向当前更新提供正确的【外力加速度方向引导】。\n\n"
        "3. 旋转越界状态偏置 (States 6-7) 含义:\n"
        "   • 单摆旋转跨越顶端 (±pi rad) 时，在相空间中会发生 ±2pi 的坐标突变。\n"
        "   • 偏置向量 b_6, b_7 的角度分量固定为 ±6.2832 (±2pi)，角速度分量为 0。\n"
        "   • 当单摆发生越界时，该状态作为【坐标重置开关】，瞬间加上 ±2pi 的偏移,\n"
        "     从而保证单摆角度始终被完美约束在 [-pi, pi] 的区间内，而速度保持平滑连续。"
    )
    
    ax_text.text(0.02, 0.95, text_content, transform=ax_text.transAxes, fontsize=10.5,
                 fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#f7f9fa', edgecolor='#d1d5db', alpha=0.9))
    
    plt.suptitle("钟摆分段线性动力学(rAR-HMM)固定偏置 $b_k$ 物理概念解析", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save results
    run_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "viz_fixed_biases_concept.png", dpi=200)
    
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
    fig.savefig(artifacts_dir / "K7_fixed_biases_concept.png", dpi=200)
    plt.close(fig)
    print(f"Saved concept plot to {run_dir / 'viz_fixed_biases_concept.png'}")
    print(f"Saved concept plot also to artifacts as K7_fixed_biases_concept.png")

if __name__ == "__main__":
    main()
