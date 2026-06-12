import numpy as np

def main():
    subj_data = np.load("data/subject_trials_preprocessed.npz", allow_pickle=True)
    x_start = subj_data["x_start"]              # (N, 2, 2)
    
    # Extract starting state of blackout
    theta0 = x_start[:, -1, 0]
    omega0 = x_start[:, -1, 1]
    
    N = len(theta0)
    
    quadrant_counts = {
        "Q1 (+, +)": np.sum((theta0 > 0) & (omega0 > 0)),
        "Q2 (-, +)": np.sum((theta0 < 0) & (omega0 > 0)),
        "Q3 (-, -)": np.sum((theta0 < 0) & (omega0 < 0)),
        "Q4 (+, -)": np.sum((theta0 > 0) & (omega0 < 0)),
        "Origin": np.sum((theta0 == 0) | (omega0 == 0))
    }
    
    print(f"Total trials: {N}")
    for quad, count in quadrant_counts.items():
        print(f"  {quad}: {count} ({count/N*100:.2f}%)")

if __name__ == "__main__":
    main()
