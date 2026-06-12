import numpy as np
from pathlib import Path
import json

def main():
    root = Path("d:/intuitive physics/pendulum_dataset/runs/hypersearch_vi")
    runs = sorted(root.glob("mid_*_high_*"))
    print(f"Found {len(runs)} hypersearch runs.")
    for run in runs:
        ll_path = run / "loglik_history.npy"
        config_path = run / "config.json"
        if ll_path.exists():
            history = np.load(ll_path)
            # Load config
            with open(config_path, "r") as f:
                cfg = json.load(f)
            print(f"Run: {run.name}")
            print(f"  Initial Log-Likelihood: {history[0]:.2f}")
            print(f"  Final Log-Likelihood: {history[-1]:.2f}")
            print(f"  Iter counts: {len(history)}")
        else:
            print(f"Run {run.name}: loglik_history.npy not found.")

if __name__ == "__main__":
    main()
