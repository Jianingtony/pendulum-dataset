import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split, stack_for_ar
from scripts.train_and_viz_k7 import build_fixed_b_k7, stratified_subset

def main():
    theta_mid = 20.0
    theta_high = 90.0
    g, L = 9.8, 4.0
    w0 = math.sqrt(g / L)
    dt = 0.05
    seed = 20260518
    
    cfg = Config(
        K=7,
        obs_repr="theta_omega",
        recurrence_mode="ro",
        L=L,
        g=g,
        init_seed=seed,
    )
    
    fixed_b = build_fixed_b_k7(theta_mid, theta_high, w0, dt=dt)
    
    # Load same training dataset
    data_root = "data/pendulum_L4/pendulum"
    trajs_all = load_split(data_root, "train", cfg, max_trajs=None)
    trajs = stratified_subset(trajs_all, 100, seed=seed)
    
    X_in, X_out, traj_idx, t_idx = stack_for_ar(trajs, P=1)
    
    theta_t = X_in[:, 0]
    theta_next = X_out[:, 0]
    diff_theta = theta_next - theta_t
    
    is_ccw_wrap = diff_theta < -5.0
    is_cw_wrap = diff_theta > 5.0
    
    anchors = np.array([0.0, np.radians(theta_mid), -np.radians(theta_mid), np.radians(theta_high), -np.radians(theta_high)])
    
    z_flat = np.zeros(len(X_out), dtype=np.int64)
    for i in range(len(X_out)):
        if is_ccw_wrap[i]:
            z_flat[i] = 5 # State 6
        elif is_cw_wrap[i]:
            z_flat[i] = 6 # State 7
        else:
            th = theta_t[i]
            z_flat[i] = np.argmin(np.abs(th - anchors))
            
    print("=== Initial Physical State Assignment Statistics ===")
    for k in range(7):
        count = np.sum(z_flat == k)
        pct = count / len(z_flat) * 100
        print(f"  State {k+1}: count = {count} ({pct:.2f}%)")
        
    print("\n=== Initial Parameter Fit ===")
    A = np.zeros((7, 2, 3))
    Q = np.zeros((7, 2, 2))
    
    for k in range(7):
        idx = (z_flat == k)
        Xk, Yk = X_in[idx], X_out[idx]
        b_k = fixed_b[k]
        
        # Regress without bias column
        Xk_no_bias = Xk[:, :-1]
        Yk_no_bias = Yk - b_k[None, :]
        Reg = 1e-4 * np.eye(Xk_no_bias.shape[1])
        A_dyn = np.linalg.solve(Xk_no_bias.T @ Xk_no_bias + Reg, Xk_no_bias.T @ Yk_no_bias).T
        B = np.concatenate([A_dyn, b_k[:, None]], axis=1)
        
        A[k] = B
        resid = Yk - Xk @ B.T
        Q[k] = resid.T @ resid / max(idx.sum() - 1, 1) + 1e-4 * np.eye(2)
        
        eigenvals = np.linalg.eigvals(A_dyn)
        print(f"  State {k+1} ({'Wrap' if k >= 5 else 'Osc'}):")
        print(f"    b_k: {b_k}")
        print(f"    A_dyn eigenvalues: {eigenvals}")
        print(f"    Eigenvalue magnitudes: {np.abs(eigenvals)}")
        print(f"    Q_k diag: {np.diag(Q[k])}")

if __name__ == "__main__":
    main()
