import sys
import numpy as np
import pickle
from pathlib import Path

# Physical constants for L=4
g = 9.8
L = 4.0
w0 = np.sqrt(g / L)

def wrap_pi(val):
    return (val + np.pi) % (2.0 * np.pi) - np.pi

def main():
    subj_data = np.load("data/subject_trials_preprocessed.npz", allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    theta_est = subj_data["theta_estimated"]    # (N,)
    energy_phys = subj_data["energy_phys"]      # (N,)
    
    # Calculate theta_max for each trial
    theta_max = []
    for E in energy_phys:
        if E >= 78.4:
            theta_max.append(np.pi)
        else:
            theta_max.append(np.arccos(1.0 - E / 39.2))
    theta_max = np.array(theta_max)
    
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from viz_hypersearch_results import rollout_deterministic_pure
    
    model_path = Path("runs/hypersearch_vi/mid_30.0_high_90.0/chain.pkl")
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["cfg"]
    p_model = ckpt["samples"][-1]
    
    errors = []
    for i in range(len(theta_est)):
        prefix_x = x_start[i]
        x_pred = rollout_deterministic_pure(cfg, p_model, prefix_x, horizon=13)
        theta_pred = x_pred[-1, 0]
        raw_err = abs(wrap_pi(theta_pred - theta_est[i]))
        norm_err = raw_err / theta_max[i]
        errors.append(norm_err)
        
    errors = np.array(errors)
    print("Normalized Errors Stats:")
    print(f"  Min: {errors.min():.4f}")
    print(f"  Max: {errors.max():.4f}")
    print(f"  Mean: {errors.mean():.4f}")
    print(f"  Median: {np.median(errors):.4f}")
    print(f"  90th percentile: {np.percentile(errors, 90):.4f}")
    print(f"  95th percentile: {np.percentile(errors, 95):.4f}")
    print(f"  98th percentile: {np.percentile(errors, 98):.4f}")

if __name__ == "__main__":
    main()
