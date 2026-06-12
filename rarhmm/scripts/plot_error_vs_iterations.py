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
from scripts.train_and_viz_k7 import rollout_deterministic_pure, wrap_pi

def main():
    run_dir = Path("runs/K7_fixed_b_vi_v3")
    subject_npz = "data/subject_trials_preprocessed.npz"
    artifacts_dir = Path(r"C:\Users\tonyj\.gemini\antigravity-ide\brain\129dffbf-6d0f-4286-886c-1b5d97144926")
    
    ckpt_path = run_dir / "chain.pkl"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found. Train the model first!")
        return
        
    print("Loading checkpoint and test dataset...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["cfg"]
    samples_history = ckpt["samples"]
    
    subj_data = np.load(subject_npz, allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    theta_est = subj_data["theta_estimated"]    # (N,)
    theta_act = subj_data["theta_actual"]       # (N,)
    N_trials = x_start.shape[0]
    
    n_iters = len(samples_history)
    print(f"Evaluating {N_trials} trials across {n_iters} iterations...")
    
    mae_history = []
    
    # Evaluate every iteration (or every step)
    for it in range(n_iters):
        p_model = samples_history[it]
        raw_errors_deg = np.zeros(N_trials)
        for i in range(N_trials):
            prefix_x = x_start[i]
            x_pred = rollout_deterministic_pure(cfg, p_model, prefix_x, horizon=13)
            theta_pred = x_pred[-1, 0]
            
            # Model's error relative to physics (theta_actual)
            model_err_rad = abs(wrap_pi(theta_pred - theta_act[i]))
            # Subject's error relative to physics
            subj_err_rad = abs(wrap_pi(theta_est[i] - theta_act[i]))
            
            # Absolute difference of these errors
            diff_err_rad = abs(model_err_rad - subj_err_rad)
            raw_errors_deg[i] = np.degrees(diff_err_rad)
            
        mae_deg = raw_errors_deg.mean()
        mae_history.append(mae_deg)
        if (it + 1) % 10 == 0 or it == 0 or it == n_iters - 1:
            print(f"  Iteration {it+1}/{n_iters}: Test MAE = {mae_deg:.3f}°")
            
    # Plotting
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(np.arange(1, n_iters + 1), mae_history, color="#d62728", lw=2, marker='o', markersize=4, label="模型与被试相对于物理真实的误差之差")
    
    # Beautify
    ax.set_xlabel("EM 迭代次数", fontsize=11, fontweight='bold')
    ax.set_ylabel("误差之差的平均绝对值 (MAE, 度)", fontsize=11, fontweight='bold')
    ax.set_title("模型与被试相对于物理真实的误差之差随训练迭代的变化曲线", fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    
    # Add annotation for start and end error
    ax.annotate(f"初始误差之差: {mae_history[0]:.2f}°", xy=(1, mae_history[0]), xytext=(10, mae_history[0] + 0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=9.5)
    ax.annotate(f"收敛误差之差: {mae_history[-1]:.2f}°", xy=(n_iters, mae_history[-1]), xytext=(n_iters - 25, mae_history[-1] + 0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=9.5)
                
    fig.tight_layout()
    
    # Save plots
    fig.savefig(run_dir / "viz_error_vs_iterations.png", dpi=200)
    fig.savefig(artifacts_dir / "K7_fixed_b_v3_error_vs_iterations.png", dpi=200)
    plt.close(fig)
    
    print(f"Saved error history plot to {run_dir / 'viz_error_vs_iterations.png'} and artifacts.")

if __name__ == "__main__":
    main()
