import pandas as pd
import numpy as np
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
    workspace_dir = Path(r"d:\intuitive physics\pendulum_dataset")
    csv_dir = workspace_dir / "subjectdata"
    stimuli_dir = csv_dir / "stimuli"
    out_dir = workspace_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = ["0004", "0006", "0007", "0008", "0009", "0010"]
    
    all_x_start = []
    all_theta_estimated = []
    all_theta_actual = []
    all_omega_actual = []
    all_energy_phys = []
    all_subject_id = []
    all_trial_id = []
    
    for sub in subjects:
        csv_path = csv_dir / f"experiment_data_subject{sub}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        print(f"\nProcessing Subject {sub}: {len(df)} trials...")
        
        # Determine stimulus JSON file name
        # Subject number modulo 5 mapped to 1-5 index
        s_num = int(sub)
        j_num = (s_num % 5) + 1
        json_path = stimuli_dir / f"stimulus-0{j_num}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Required stimulus file {json_path} not found.")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            stim_data = json.load(f)
            
        # Extract all physics trials from JSON
        json_trials = []
        for block in stim_data["sequence"]:
            if block.get("kind") in ["block", "practice"]:
                for trial in block.get("children", []):
                    phys_unit = None
                    for unit in trial.get("units", []):
                        if unit.get("type") in ["pendulumStimulus", "pendulumPractice"]:
                            phys_unit = unit
                            break
                    if phys_unit is not None:
                        json_trials.append({
                            "trial_id": trial["id"],
                            "theta0": math.radians(phys_unit["theta0Deg"]),
                            "omega0": math.radians(phys_unit["omega0DegPerSec"]),
                        })
        print(f"  Loaded JSON {json_path.name} with {len(json_trials)} trials.")
        
        # Match each CSV trial to a JSON trial
        matched_count = 0
        for i in range(len(df)):
            row = df.iloc[i]
            E_sub = float(row["pendulum_E_J"])
            theta_act = float(row["theta_actual_rad"])
            theta_est = float(row["theta_estimated_rad"])
            t_end = float(row["total_time_sec"])
            
            # Find the best matching trial in JSON
            best_jt = None
            best_diff = 1e9
            best_omega_sim = 0.0
            
            for jt in json_trials:
                # Energy check
                E_bar_jt = 0.5 * (jt["omega0"]**2) / w0_sq + 1.0 - math.cos(jt["theta0"])
                E_phys_jt = (g * L) * E_bar_jt
                
                if abs(E_phys_jt - E_sub) < 0.1:
                    # Run simulation to check terminal angle
                    sol = solve_ivp(pendulum_deriv, [0, t_end], [jt["theta0"], jt["omega0"]], t_eval=[t_end], rtol=1e-8, atol=1e-8)
                    theta_sim = sol.y[0][0]
                    diff = abs(wrap_pi(theta_sim - theta_act))
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_jt = jt
                        best_omega_sim = sol.y[1][0]
            
            # Allow matching threshold
            if best_diff < 0.1: # Less than ~5 degrees error
                matched_count += 1
                
                # We have found the matched JSON trial, now compute the blackout start state
                # Blackout time duration is exactly 0.65 seconds (fade_sec = 0.15 + hide_sec = 0.5)
                # We extract state at t_start - 0.05 AND t_start to provide a prefix of length 2 >= P (for FFBS)
                t_start = t_end - 0.65
                t_eval_pts = [t_start - 0.05, t_start]
                
                sol_start = solve_ivp(pendulum_deriv, [0, t_start], [best_jt["theta0"], best_jt["omega0"]], t_eval=t_eval_pts, rtol=1e-8, atol=1e-8)
                
                theta_pre = sol_start.y[0][0]
                omega_pre = sol_start.y[1][0]
                theta_start = sol_start.y[0][1]
                omega_start = sol_start.y[1][1]
                
                # Normalize omega by w0
                omega_pre_normalized = omega_pre / w0
                omega_start_normalized = omega_start / w0
                
                # x_prefix shape is (2, 2)
                x_prefix = np.array([
                    [theta_pre, omega_pre_normalized],
                    [theta_start, omega_start_normalized]
                ], dtype=np.float64)
                
                all_x_start.append(x_prefix)
                all_theta_estimated.append(theta_est)
                all_theta_actual.append(theta_act)
                all_omega_actual.append(best_omega_sim)
                all_energy_phys.append(E_sub)
                all_subject_id.append(sub)
                all_trial_id.append(best_jt["trial_id"])
            else:
                print(f"  Warning: Trial {i} in Subject {sub} could not be matched. Min diff: {best_diff:.4f}")
                
        print(f"  Successfully matched {matched_count} / {len(df)} trials.")
        
    # Convert to numpy arrays
    x_start_arr = np.array(all_x_start, dtype=np.float64)
    theta_est_arr = np.array(all_theta_estimated, dtype=np.float64)
    theta_act_arr = np.array(all_theta_actual, dtype=np.float64)
    omega_act_arr = np.array(all_omega_actual, dtype=np.float64)
    energy_phys_arr = np.array(all_energy_phys, dtype=np.float64)
    subject_id_arr = np.array(all_subject_id, dtype=object)
    trial_id_arr = np.array(all_trial_id, dtype=object)
    
    # Save as NPZ
    npz_out_path = out_dir / "subject_trials_preprocessed.npz"
    np.savez(
        npz_out_path,
        x_start=x_start_arr,
        theta_estimated=theta_est_arr,
        theta_actual=theta_act_arr,
        omega_actual=omega_act_arr,
        energy_phys=energy_phys_arr,
        subject_id=subject_id_arr,
        trial_id=trial_id_arr
    )
    print(f"\nSaved preprocessed trials to {npz_out_path}")
    print(f"Total processed trials: {len(all_x_start)}")

if __name__ == "__main__":
    main()
