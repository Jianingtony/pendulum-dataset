import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle
import numpy as np

def inspect_run(run_name, path):
    print(f"\n======================================")
    print(f"Run: {run_name}")
    try:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        samples = ckpt["samples"]
        cfg = ckpt["cfg"]
        A_mean = np.mean([s.A for s in samples], axis=0)
        Q_mean = np.mean([s.Q for s in samples], axis=0)
        z_last = ckpt["z_last"]
        
        counts = np.bincount(np.concatenate(z_last), minlength=cfg.K) if len(z_last) > 0 else []
        print(f"Counts per state: {counts}")
        
        for k in range(cfg.K):
            print(f"  State {k+1}:")
            print(f"    b_k: {A_mean[k, :, -1]}")
            print(f"    Q_k diag: {np.diag(Q_mean[k])}")
            print(f"    Q_k:\n{Q_mean[k]}")
            print(f"    A_dyn:\n{A_mean[k, :, :-1]}")
    except Exception as e:
        print(f"Error reading {run_name}: {e}")

inspect_run("K5_fixed_b_vi", "d:/intuitive physics/pendulum_dataset/rarhmm/runs/K5_fixed_b_vi/chain.pkl")
inspect_run("K5_theta_allE_vi", "d:/intuitive physics/pendulum_dataset/rarhmm/runs/K5_theta_allE_vi/chain.pkl")
inspect_run("K10_theta_allE_wrap_vi", "d:/intuitive physics/pendulum_dataset/rarhmm/runs/K10_theta_allE_wrap_vi/chain.pkl")
