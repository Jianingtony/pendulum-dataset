import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import numpy as np
from rarhmm.data import load_split

runs_dir = Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs")
for run_path in sorted(runs_dir.glob("**/chain.pkl")):
    with open(run_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["cfg"]
    z_last = ckpt["z_last"]
    samples = ckpt["samples"]
    print(f"\n======================================")
    print(f"Run: {run_path.parent.name}")
    print(f"K={cfg.K}, obs_repr={cfg.obs_repr}, recurrence_mode={cfg.recurrence_mode}")
    
    A_mean = np.mean([s.A for s in samples], axis=0)
    
    trajs = load_split("d:/intuitive physics/pendulum_dataset/data/pendulum", "train", cfg, max_trajs=len(z_last))
    state_points = {k: [] for k in range(cfg.K)}
    for i, tr in enumerate(trajs):
        z_i = z_last[i]
        T_i = tr.x.shape[0]
        for t in range(min(T_i, len(z_i))):
            zt = z_i[t]
            state_points[zt].append((tr.theta[t], tr.omega[t]))
            
    for k in range(cfg.K):
        pts = state_points[k]
        if len(pts) > 0:
            pts = np.array(pts)
            mean_theta = np.mean(pts[:, 0])
            std_theta = np.std(pts[:, 0])
            mean_omega = np.mean(pts[:, 1])
            std_omega = np.std(pts[:, 1])
            bk = A_mean[k][:, -1]
            print(f"  State {k+1}: n={len(pts):5d} | theta = {mean_theta:+.3f} +- {std_theta:.3f} | omega = {mean_omega:+.3f} +- {std_omega:.3f} | b_k = {bk}")
        else:
            print(f"  State {k+1}: empty")
