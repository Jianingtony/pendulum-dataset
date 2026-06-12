import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split
from scripts.train_and_viz_k7 import build_fixed_b_k7, stratified_subset
from rarhmm.model import RecurrentARHMM
from rarhmm.inference import initialize

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
    
    data_root = "data/pendulum_L4/pendulum"
    trajs_all = load_split(data_root, "train", cfg, max_trajs=None)
    trajs = stratified_subset(trajs_all, 100, seed=seed)
    
    model = RecurrentARHMM(cfg)
    
    rng = np.random.default_rng(seed)
    z_per = initialize(model, trajs, rng, fixed_b=fixed_b)
    
    print("=== Physical Seeding ===")
    z_flat = np.concatenate(z_per)
    for k in range(7):
        print(f"  State {k+1}: count = {np.sum(z_flat == k)}")
        
    print("\n=== Fitted Recurrence Parameters (R, r) ===")
    p = model.params
    for j in range(6):
        # In 'ro' mode, R is (1, 6, 2) and r is (1, 6)
        print(f"  Stick {j+1}: R = {p.R[0, j]}, r = {p.r[0, j]:.3f}")
        
    from rarhmm.stick_breaking import stick_breaking_log_probs
    # Verify partition on a grid of angles
    print("\n=== Gating Probabilities for Angle Grid (velocity = 0) ===")
    test_angles = np.radians([-90, -45, -20, 0, 20, 45, 90])
    for ta in test_angles:
        x_test = np.array([[ta, 0.0]])
        z_dummy = np.zeros(1, dtype=np.int64)
        nu = p.recurrence_logits(x_test, z_dummy)
        log_pi = stick_breaking_log_probs(nu)[0]
        probs = np.exp(log_pi)
        prob_str = ", ".join([f"S{k+1}: {probs[k]:.1%}" for k in range(7)])
        print(f"  Angle {np.degrees(ta): 4.0f}°: {prob_str}")

if __name__ == "__main__":
    main()
