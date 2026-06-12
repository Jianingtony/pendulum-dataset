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
# Get the last sample or mean of samples
A_list = [s.A for s in samples]
Q_list = [s.Q for s in samples]
R_list = [s.R for s in samples]
r_list = [s.r for s in samples]

A_mean = np.mean(A_list, axis=0)
Q_mean = np.mean(Q_list, axis=0)
R_mean = np.mean(R_list, axis=0)
r_mean = np.mean(r_list, axis=0)

K = cfg.K
M = cfg.obs_dim
print(f"K={K}, M={M}, obs_repr={cfg.obs_repr}")
print("A_mean shape:", A_mean.shape)

# Let's print each state's dynamic matrix and bias column
for k in range(K):
    Ak = A_mean[k]
    # The bias column is Ak[:, M]
    bk = Ak[:, M]
    # The dynamics columns
    A_dyn = Ak[:, :M]
    print(f"\nState {k+1}:")
    print(f"  Bias b_k: {bk}")
    print(f"  A_dyn:\n{A_dyn}")

# Let's check where this state is mostly active in the training data
z_last = ckpt["z_last"]
# Load trajectories to match
from rarhmm.data import load_split
trajs = load_split("d:/intuitive physics/pendulum_dataset/data/pendulum", "train", cfg, max_trajs=len(z_last))

# For each state, find the mean theta and omega of points assigned to it
state_points = {k: [] for k in range(K)}
for i, tr in enumerate(trajs):
    z_i = z_last[i]
    T_i = tr.x.shape[0]
    for t in range(min(T_i, len(z_i))):
        zt = z_i[t]
        # Get actual theta and omega
        state_points[zt].append((tr.theta[t], tr.omega[t]))

print("\nState statistics in training data:")
for k in range(K):
    pts = state_points[k]
    if len(pts) > 0:
        pts = np.array(pts)
        mean_theta = np.mean(pts[:, 0])
        std_theta = np.std(pts[:, 0])
        mean_omega = np.mean(pts[:, 1])
        std_omega = np.std(pts[:, 1])
        print(f"State {k+1}: n={len(pts)}, theta = {mean_theta:+.4f} +- {std_theta:.4f} rad ({np.degrees(mean_theta):+.1f} deg), omega = {mean_omega:+.4f} +- {std_omega:.4f} rad/s")
    else:
        print(f"State {k+1}: empty")
