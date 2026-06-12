import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import numpy as np


run_path = Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs/K10_theta_allE_wrap_vi/chain.pkl")
with open(run_path, "rb") as f:
    ckpt = pickle.load(f)

cfg = ckpt["cfg"]
samples = ckpt["samples"]
A_mean = np.mean([s.A for s in samples], axis=0)
K = cfg.K
M = cfg.obs_dim
omega0 = cfg.omega0

print(f"K={K}, M={M}, omega0={omega0:.4f}")
print("Computing physical centers of states (I - A_dyn)^-1 * b:")

for k in range(K):
    Ak = A_mean[k]
    bk = Ak[:, M]
    A_dyn = Ak[:, :M]
    
    # Solve (I - A_dyn) * x = b
    try:
        x_center = np.linalg.solve(np.eye(M) - A_dyn, bk)
        theta_center = x_center[0]
        omega_center = x_center[1] * omega0
        print(f"State {k+1}:")
        print(f"  b_k: {bk}")
        print(f"  Center: theta = {theta_center:+.4f} rad ({np.degrees(theta_center):+.1f} deg), omega = {omega_center:+.4f} rad/s")
    except np.linalg.LinAlgError:
        print(f"State {k+1}: Singular A_dyn, no unique center")
