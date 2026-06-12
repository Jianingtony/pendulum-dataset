import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rarhmm.config import Config
from rarhmm.data import load_split

def main():
    cfg = Config(
        K=7,
        obs_repr="theta_omega",
        recurrence_mode="ro",
        L=4.0,
        g=9.8,
        init_seed=20260518,
    )
    
    val_trajs = load_split("data/pendulum_L4/pendulum", "val", cfg)
    print(f"Loaded {len(val_trajs)} validation trajectories.")
    
    large_angle_trajs = []
    rotation_trajs = []
    
    for tr in val_trajs:
        max_theta = np.max(np.abs(tr.theta))
        max_omega = np.max(np.abs(tr.omega))
        
        # Check if it wraps/rotates or has large angle
        if tr.regime == "rotation" or np.any(np.abs(tr.theta) > 2.8):
            rotation_trajs.append((tr.id, max_theta, max_omega, tr.regime))
        elif max_theta > 1.5:
            large_angle_trajs.append((tr.id, max_theta, max_omega, tr.regime))
            
    print(f"\nFound {len(rotation_trajs)} rotation/wrap-around trajectories:")
    for tid, mt, mo, reg in rotation_trajs[:10]:
        print(f"  ID: {tid}, max theta: {mt:.2f}, max omega: {mo:.2f}, regime: {reg}")
        
    print(f"\nFound {len(large_angle_trajs)} large-angle swing trajectories:")
    for tid, mt, mo, reg in large_angle_trajs[:10]:
        print(f"  ID: {tid}, max theta: {mt:.2f}, max omega: {mo:.2f}, regime: {reg}")

if __name__ == "__main__":
    main()
