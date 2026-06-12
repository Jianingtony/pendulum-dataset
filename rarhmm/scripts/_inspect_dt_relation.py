import numpy as np
from pathlib import Path

def main():
    path = Path("d:/intuitive physics/pendulum_dataset/data/pendulum_L4/pendulum/train.npz")
    data = np.load(path, allow_pickle=True)
    
    # Let's inspect the first trajectory
    theta = data["theta"][0]
    omega = data["omega"][0]
    
    print(f"L4 training data trajectory 0:")
    print(f"  theta shape: {theta.shape}")
    print(f"  omega shape: {omega.shape}")
    
    # Let's compute delta theta: theta[t] - theta[t-1]
    d_theta = np.diff(theta)
    
    # Let's check the relationship between d_theta and omega[t-1]
    # In physics: d_theta_t \approx omega_{t-1} * dt
    dt = 0.05
    ratio = d_theta / (omega[:-1] * dt)
    print(f"  Mean ratio (d_theta / (omega_prev * dt)): {np.mean(ratio):.6f}")
    print(f"  First 5 d_theta: {d_theta[:5]}")
    print(f"  First 5 omega_prev * dt: {omega[:-1][:5] * dt}")
    
    # Now let's check the scaling of omega in the model input x
    # omega0 = sqrt(9.8/4.0) = 1.565247584
    omega0 = np.sqrt(9.8 / 4.0)
    omega_norm = omega / omega0
    
    # In x: x[t, 0] = theta[t], x[t, 1] = omega[t] / omega0
    # The model transition is: theta_t \approx A[0,0] theta_{t-1} + A[0,1] omega_norm_{t-1} + b_theta
    # If A[0,0] \approx 1, then theta_t - theta_{t-1} \approx A[0,1] omega_norm_{t-1} + b_theta
    # Since omega_norm = omega / omega0, we have:
    # theta_t - theta_{t-1} \approx A[0,1] (omega_{t-1} / omega0)
    # But from physics: theta_t - theta_{t-1} \approx omega_{t-1} * dt
    # So: A[0,1] / omega0 \approx dt  ==>  A[0,1] \approx omega0 * dt
    # Let's check: omega0 * dt = 1.565247584 * 0.05 = 0.078262
    # But the model learned A[0,1] \approx 0.1565!
    # 0.1565 / 0.078262 \approx 2.0 !!!
    # Why is it 2.0? Let's check the step-wise change in theta in the dataset!
    
if __name__ == "__main__":
    main()
