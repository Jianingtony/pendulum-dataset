import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split
from scripts.train_and_viz_k7 import stratified_subset

def main():
    cfg = Config(
        K=7,
        obs_repr="theta_omega",
        recurrence_mode="ro",
        L=4.0,
        g=9.8,
        init_seed=20260518,
    )
    
    # Load same training dataset used in K7 V2
    data_root = "data/pendulum_L4/pendulum"
    trajs_all = load_split(data_root, "train", cfg, max_trajs=None)
    trajs = stratified_subset(trajs_all, 100, seed=20260518)
    
    print(f"Loaded {len(trajs)} stratified training trajectories.")
    
    all_theta = []
    all_omega = []
    
    ccw_wraps = 0
    cw_wraps = 0
    
    for tr in trajs:
        all_theta.extend(tr.theta)
        all_omega.extend(tr.omega)
        
        # Detect wrap-arounds in raw data
        # In raw theta (unwrapped), a rotation crossing pi is visible as a jump
        for t in range(1, len(tr.theta)):
            diff = tr.theta[t] - tr.theta[t-1]
            # If wrapped to [-pi, pi] was used, a jump from approx pi to -pi or vice versa:
            # Let's check diff on wrapped angles.
            # If theta_t is near -pi and theta_{t-1} is near pi, it's a CCW wrap (positive velocity, theta wrapped jumps by -2pi)
            # wait, if theta is wrapped:
            # CCW wrap: theta goes from +pi to -pi (jump of -2pi in wrapped coordinate)
            # CW wrap: theta goes from -pi to +pi (jump of +2pi in wrapped coordinate)
            if diff < -5.5:
                ccw_wraps += 1
            elif diff > 5.5:
                cw_wraps += 1
                
    all_theta = np.array(all_theta)
    all_omega = np.array(all_omega)
    
    print("\n=== Dataset Symmetry Analysis ===")
    print(f"Total time steps: {len(all_theta)}")
    print(f"Mean theta: {np.mean(all_theta):.4f} rad")
    print(f"Mean omega: {np.mean(all_omega):.4f} rad/s")
    
    pos_theta_count = np.sum(all_theta > 0)
    neg_theta_count = np.sum(all_theta < 0)
    print(f"Time steps with theta > 0: {pos_theta_count} ({pos_theta_count / len(all_theta) * 100:.2f}%)")
    print(f"Time steps with theta < 0: {neg_theta_count} ({neg_theta_count / len(all_theta) * 100:.2f}%)")
    
    pos_omega_count = np.sum(all_omega > 0)
    neg_omega_count = np.sum(all_omega < 0)
    print(f"Time steps with omega > 0: {pos_omega_count} ({pos_omega_count / len(all_theta) * 100:.2f}%)")
    print(f"Time steps with omega < 0: {neg_omega_count} ({neg_omega_count / len(all_theta) * 100:.2f}%)")
    
    print(f"\nWrap-arounds detected in wrapped coordinate space:")
    print(f"  CCW wraps (+pi -> -pi jumps, positive velocity): {ccw_wraps}")
    print(f"  CW wraps (-pi -> +pi jumps, negative velocity): {cw_wraps}")
    
    # Also count time steps in large angles (> 60 degrees / 1.047 rad)
    large_angle_threshold = 1.047
    large_pos_theta = np.sum(all_theta > large_angle_threshold)
    large_neg_theta = np.sum(all_theta < -large_angle_threshold)
    print(f"\nTime steps in high angles (|theta| > 60 degrees):")
    print(f"  Theta > +60 deg: {large_pos_theta} ({large_pos_theta / len(all_theta) * 100:.2f}%)")
    print(f"  Theta < -60 deg: {large_neg_theta} ({large_neg_theta / len(all_theta) * 100:.2f}%)")

if __name__ == "__main__":
    main()
