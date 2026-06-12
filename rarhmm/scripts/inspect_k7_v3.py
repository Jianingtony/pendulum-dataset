import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path("d:/intuitive physics/pendulum_dataset/rarhmm").resolve()))

def main():
    model_path = Path("d:/intuitive physics/pendulum_dataset/runs/K7_fixed_b_vi_v3/chain.pkl")
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return
        
    with open(model_path, "rb") as f:
        ckpt = pickle.load(model_path) if hasattr(pickle, "load_f") else pickle.load(f)
        
    p_model = ckpt["samples"][-1]
    
    print("=== K7 V3 Model Parameters ===")
    print("Recurrence logits weights R (shape K x K-1 x M):")
    print(p_model.R[0])
    print("Recurrence logits bias r (shape K x K-1):")
    print(p_model.r[0])
    
    print("\nState Biases b (from A matrix last column):")
    for k in range(7):
        b_k = p_model.A[k, :, 2]
        print(f"  State {k+1}: b_{k+1} = {b_k}")
        
    print("\nDynamics transition matrices A_k (upper 2x2):")
    for k in range(7):
        A_k = p_model.A[k, :, :2]
        print(f"  State {k+1}:")
        print(A_k)
        
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(A_k)
        print(f"    Eigenvalues: {eigenvals}")
        print(f"    Eigenvalue magnitudes: {np.abs(eigenvals)}")
        
    print("\nCovariances Q_k:")
    for k in range(7):
        print(f"  State {k+1}:")
        print(p_model.Q[k])

if __name__ == "__main__":
    main()
