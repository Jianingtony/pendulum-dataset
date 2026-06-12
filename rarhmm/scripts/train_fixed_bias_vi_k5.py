import argparse
import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split
from rarhmm.train_vi import fit_vi

def build_fixed_b(theta_mid_deg, theta_high_deg, w0, dt=0.05, alpha=0.08):
    """Build the constrained (5, 2) bias matrix b based on mid/high angles."""
    theta_mid_rad = math.radians(theta_mid_deg)
    theta_high_rad = math.radians(theta_high_deg)
    
    C1 = 0.5 * w0 * dt  # physical integration constant (approx 0.03913 for L=4)
    
    b1 = np.array([0.0, 0.0])
    b2 = np.array([C1 * alpha * theta_mid_rad, alpha * theta_mid_rad])
    b3 = np.array([-C1 * alpha * theta_mid_rad, -alpha * theta_mid_rad])
    b4 = np.array([C1 * alpha * theta_high_rad, alpha * theta_high_rad])
    b5 = np.array([-C1 * alpha * theta_high_rad, -alpha * theta_high_rad])
    
    return np.vstack([b1, b2, b3, b4, b5])

def stratified_subset(trajs, target_n: int, seed: int):
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    by_E = defaultdict(list)
    for i, tr in enumerate(trajs):
        by_E[round(tr.E_bar, 6)].append(i)

    picked, pool = [], []
    for E, idxs in by_E.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        picked.append(idxs[0])
        pool.extend(idxs[1:])

    n_bins = len(by_E)
    if target_n < n_bins:
        rng.shuffle(picked)
        picked = picked[:target_n]
    else:
        remainder = target_n - n_bins
        if remainder > 0:
            rng.shuffle(pool)
            picked.extend(pool[:remainder])

    picked.sort(key=lambda i: trajs[i].E_bar)
    return [trajs[i] for i in picked]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/pendulum_L4/pendulum")
    p.add_argument("--out", type=str, default="runs/K5_fixed_b_vi")
    p.add_argument("--theta-mid", type=float, default=30.0, help="Mid state angle HP in degrees.")
    p.add_argument("--theta-high", type=float, default=90.0, help="High state angle HP in degrees.")
    p.add_argument("--n-em-iter", type=int, default=80)
    p.add_argument("--n-r-steps", type=int, default=100)
    p.add_argument("--r-lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--target-n", type=int, default=100, help="Subset size for training speed.")
    args = p.parse_args()
    
    cfg = Config(
        K=5,
        obs_repr="theta_omega",
        recurrence_mode="ro",
        L=4.0,
        g=9.8,
        init_seed=args.seed,
        out_dir=args.out,
    )
    
    # Build fixed bias matrix b
    w0 = cfg.omega0  # w0 = sqrt(9.8/4.0) = 1.5652476
    fixed_b = build_fixed_b(args.theta_mid, args.theta_high, w0, dt=cfg.dt)
    print(f"Built (5, 2) fixed_b matrix for theta_mid={args.theta_mid} deg, theta_high={args.theta_high} deg:")
    print(fixed_b)
    
    # Load L=4.0 training dataset
    trajs_all = load_split(args.data_root, "train", cfg, max_trajs=None)
    print(f"Loaded {len(trajs_all)} train trajectories.")
    if args.target_n is not None and args.target_n < len(trajs_all):
        trajs = stratified_subset(trajs_all, args.target_n, seed=args.seed)
        print(f"Selected stratified subset of {len(trajs)} trajectories.")
    else:
        trajs = trajs_all
        
    # Fit model with fixed bias constraint
    fit_vi(
        cfg,
        trajs,
        n_em_iter=args.n_em_iter,
        n_r_steps=args.n_r_steps,
        r_lr=args.r_lr,
        verbose=True,
        fixed_b=fixed_b
    )

if __name__ == "__main__":
    main()
