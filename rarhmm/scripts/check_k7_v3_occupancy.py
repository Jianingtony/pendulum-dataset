import sys
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split
from rarhmm.inference import _per_traj_logobs_logtrans, ffbs_single
from scripts.train_and_viz_k7 import stratified_subset

def main():
    run_path = Path(r"d:\intuitive physics\pendulum_dataset\runs\K7_fixed_b_vi_v3\chain.pkl")
    if not run_path.exists():
        print(f"Error: {run_path} does not exist.")
        return
        
    with open(run_path, "rb") as f:
        ckpt = pickle.load(f)
        
    cfg: Config = ckpt["cfg"]
    samples = ckpt["samples"]
    p = samples[-1]  # Converged model parameters (iteration 100)
    
    # Load same training data subset
    data_root = "data/pendulum_L4/pendulum"
    seed = 20260518
    target_n = 100
    
    trajs_all = load_split(data_root, "train", cfg, max_trajs=None)
    trajs = stratified_subset(trajs_all, target_n, seed=seed)
    print(f"Loaded {len(trajs)} stratified train trajectories.")
    
    # Count state occurrences
    state_counts = np.zeros(cfg.K)
    rng = np.random.default_rng(42)
    
    log_init = ckpt.get("log_init", np.full(cfg.K, -np.log(cfg.K)))
    
    total_points = 0
    for i, tr in enumerate(trajs):
        T = tr.x.shape[0]
        if T <= cfg.ar_lag:
            continue
        bundle = _per_traj_logobs_logtrans(tr, p, cfg)
        if bundle is None:
            continue
        log_obs, log_trans, _ = bundle
        z_sampled = ffbs_single(log_init, log_trans, log_obs, rng)
        for z in z_sampled:
            state_counts[z] += 1
            total_points += 1
            
    print(f"\nEmpirical State Occupancy on 100 training trajectories for V3 ({total_points} total points):")
    for k in range(cfg.K):
        count = state_counts[k]
        percentage = 100.0 * count / total_points if total_points > 0 else 0
        print(f"  State {k+1}: count = {int(count):d} ({percentage:.2f}%)")

if __name__ == '__main__':
    main()
