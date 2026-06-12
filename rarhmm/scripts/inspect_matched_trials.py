import numpy as np
import pandas as pd
import json
import math
from pathlib import Path
from scipy.integrate import solve_ivp

# Physical Constants for L=4
g = 9.8
L = 4.0
w0 = math.sqrt(g / L)
w0_sq = g / L

def pendulum_deriv(t, y):
    theta, omega = y
    return [omega, -w0_sq * math.sin(theta)]

def wrap_pi(val):
    return (val + np.pi) % (2.0 * np.pi) - np.pi

def main():
    subj_data = np.load("data/subject_trials_preprocessed.npz", allow_pickle=True)
    x_start = subj_data["x_start"]
    theta_estimated = subj_data["theta_estimated"]
    theta_actual = subj_data["theta_actual"]
    subject_id = subj_data["subject_id"]
    trial_id = subj_data["trial_id"]
    
    print("Inspecting first 15 preprocessed trials:")
    print("---------------------------------------")
    for i in range(15):
        # x_start[i, -1] is the state at t_start
        th_start = x_start[i, -1, 0]
        om_start_norm = x_start[i, -1, 1]
        om_start = om_start_norm * w0
        
        # Check if same sign or opposite sign
        sign_str = "Same Sign (Rising)" if th_start * om_start > 0 else "Opposite Sign (Falling)"
        if th_start == 0 or om_start == 0:
            sign_str = "Zero crossing / Turnaround"
            
        print(f"Trial {i+1} (Sub: {subject_id[i]}, Trial ID: {trial_id[i]}):")
        print(f"  Start: theta_0 = {th_start:+.4f} rad ({np.degrees(th_start):+.1f}°), omega_0 = {om_start:+.4f} rad/s")
        print(f"  {sign_str}")
        print(f"  Terminal actual angle: {theta_actual[i]:+.4f} rad ({np.degrees(theta_actual[i]):+.1f}°)")

if __name__ == "__main__":
    main()
