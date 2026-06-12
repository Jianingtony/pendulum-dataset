import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import numpy as np

def main():
    run_path = Path("d:/intuitive physics/pendulum_dataset/rarhmm/runs/K10_theta_allE_wrap_vi/chain.pkl")
    with open(run_path, "rb") as f:
        ckpt = pickle.load(f)
    
    samples = ckpt["samples"]
    p = samples[-1]  # Get final parameters
    
    print("Recurrence Parameters:")
    R = p.R[0]  # (K-1, M)
    r = p.r[0]  # (K-1,)
    print(f"R (K-1, M):\n{R}")
    print(f"r (K-1,):\n{r}")
    
    # Let's inspect the transition probability function stick_breaking_log_probs
    from rarhmm.stick_breaking import stick_breaking_log_probs
    
    # We want to understand what the unconstrained model is doing.
    # Specifically, why did the unconstrained model fit the physics so well?
    # Let's check how the states 3,4,5,6,7,8,9,10 are active in the phase space.
    # Let's evaluate the state probabilities on a grid.
    thetas = np.linspace(-np.pi, np.pi, 5)
    omegas = np.linspace(-3.0, 3.0, 5)
    print("\nState probability grid:")
    for th in thetas:
        for om in omegas:
            x = np.array([[th, om]])
            # Compute logits
            nu = x @ R.T + r
            log_pi = stick_breaking_log_probs(nu)[0]
            pi = np.exp(log_pi)
            # Find the state with max prob
            max_z = np.argmax(pi)
            print(f"theta={th:+.2f}, omega={om:+.2f} | max state={max_z+1} (prob={pi[max_z]:.2f}) | probs={', '.join(f'{p:.2f}' for p in pi)}")

if __name__ == "__main__":
    main()
