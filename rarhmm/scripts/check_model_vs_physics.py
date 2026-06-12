import sys
import math
import pickle
import numpy as np
from pathlib import Path

# Insert parent dir to import rarhmm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import Trajectory
from rarhmm.model import ModelParams
from rarhmm.inference import _per_traj_logobs_logtrans
from rarhmm.train_vi import _forward_backward_ro
from rarhmm.stick_breaking import stick_breaking_log_probs
from scripts.viz_hypersearch_results import rollout_deterministic_pure, wrap_pi

def main():
    # Load subject data
    subj_data = np.load("data/subject_trials_preprocessed.npz", allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    theta_est = subj_data["theta_estimated"]    # (N,)
    theta_act = subj_data["theta_actual"]       # (N,)
    N = x_start.shape[0]
    
    # Load model from mid_30.0_high_90.0
    model_path = Path("runs/hypersearch_vi/mid_30.0_high_90.0/chain.pkl")
    if not model_path.exists():
        model_path = Path("d:/intuitive physics/pendulum_dataset/runs/hypersearch_vi/mid_30.0_high_90.0/chain.pkl")
        
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
        
    cfg = ckpt["cfg"]
    p_model = ckpt["samples"][-1]
    
    model_errors_vs_subj = []
    model_errors_vs_phys = []
    subj_errors_vs_phys = []
    
    for i in range(N):
        prefix_x = x_start[i]
        x_pred = rollout_deterministic_pure(cfg, p_model, prefix_x, horizon=13)
        theta_pred = x_pred[-1, 0]
        
        err_vs_subj = abs(wrap_pi(theta_pred - theta_est[i]))
        err_vs_phys = abs(wrap_pi(theta_pred - theta_act[i]))
        subj_err_vs_phys = abs(wrap_pi(theta_est[i] - theta_act[i]))
        
        model_errors_vs_subj.append(err_vs_subj)
        model_errors_vs_phys.append(err_vs_phys)
        subj_errors_vs_phys.append(subj_err_vs_phys)
        
    model_errors_vs_subj = np.array(model_errors_vs_subj)
    model_errors_vs_phys = np.array(model_errors_vs_phys)
    subj_errors_vs_phys = np.array(subj_errors_vs_phys)
    
    print("Evaluation results (in degrees):")
    print(f"  Model vs Subject MAE: {np.degrees(model_errors_vs_subj.mean()):.2f}°")
    print(f"  Model vs Physics MAE: {np.degrees(model_errors_vs_phys.mean()):.2f}°")
    print(f"  Subject vs Physics MAE: {np.degrees(subj_errors_vs_phys.mean()):.2f}°")
    print(f"  Model vs Physics Max Error: {np.degrees(model_errors_vs_phys.max()):.2f}°")

if __name__ == "__main__":
    main()
