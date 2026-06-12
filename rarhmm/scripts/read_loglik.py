import numpy as np
from pathlib import Path

def main():
    runs = {
        "K10_theta_allE_wrap_vi": Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs/K10_theta_allE_wrap_vi/loglik_history.npy"),
        "K5_fixed_b_vi": Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs/K5_fixed_b_vi/loglik_history.npy"),
        "K5_theta_allE_vi": Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs/K5_theta_allE_vi/loglik_history.npy")
    }
    
    for name, path in runs.items():
        if not path.exists():
            # Try alternate path
            path = Path("rarhmm/runs") / name / "loglik_history.npy"
        if path.exists():
            history = np.load(path)
            print(f"{name}:")
            print(f"  Shape: {history.shape}")
            print(f"  Initial Log-Likelihood: {history[0]:.4f}")
            print(f"  Final Log-Likelihood: {history[-1]:.4f}")
            print(f"  Max Log-Likelihood: {history.max():.4f}")
            print(f"  Min Log-Likelihood: {history.min():.4f}")
        else:
            print(f"{name}: Not found at {path}")

if __name__ == "__main__":
    main()
