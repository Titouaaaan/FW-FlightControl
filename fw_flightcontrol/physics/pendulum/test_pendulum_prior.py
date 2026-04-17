import torch
import numpy as np
import gymnasium as gym
import math
import sys
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')
from pendulum_physics import PendulumPhysics
from torchdiffeq import odeint


def observation_to_state(obs, previous_theta=None):
    """
    Decode Gym Pendulum-v1 observation to raw state.
    
    Observation: [cos(θ), sin(θ), ω]
    State: [θ, ω]
    
    Handles angle wrapping by tracking previous angle.
    """
    cos_theta, sin_theta, omega = obs[0], obs[1], obs[2]
    theta_wrapped = np.arctan2(sin_theta, cos_theta)  # In [-π, π]
    
    # If we have a previous angle, unwrap to maintain continuity
    if previous_theta is not None:
        # Find the wrapping offset
        diff = theta_wrapped - previous_theta
        # If diff is large, we wrapped
        if diff > np.pi:
            theta_wrapped -= 2 * np.pi
        elif diff < -np.pi:
            theta_wrapped += 2 * np.pi
    
    return np.array([theta_wrapped, omega], dtype=np.float32)


def generate_gym_trajectory(env, initial_obs, num_steps):
    """
    Run Gym's Pendulum-v1 natively for num_steps, working with observations.
    Decodes observations to states for comparison with physics prior.
    Handles angle wrapping via continuous tracking.
    
    Args:
        env: Gym environment
        initial_obs: Initial observation [cos(θ), sin(θ), ω]
        num_steps: Number of steps to run
    
    Returns:
        Trajectory of states: (num_steps+1, batch_size, 2)
    """
    trajectory = []
    device = torch.device('cpu')
    
    # Decode initial observation to state
    initial_state = observation_to_state(initial_obs, previous_theta=None)
    trajectory.append(torch.tensor([initial_state], dtype=torch.float32, device=device))
    
    # Reset env and set to initial state
    env.reset()
    env.unwrapped.state = initial_state.copy()
    
    previous_theta = initial_state[0]  # Track for angle unwrapping
    
    for _ in range(num_steps):
        action = np.array([0.0])  # Zero action
        obs, _, _, _, _ = env.step(action)
        # Decode observation back to state with unwrapping
        state = observation_to_state(obs, previous_theta=previous_theta)
        previous_theta = state[0]  # Update for next iteration
        trajectory.append(torch.tensor([state], dtype=torch.float32, device=device))
    
    return torch.stack(trajectory)


def generate_prior_trajectory_semiimplicit_euler_autodiff(physics_prior, initial_state, num_steps, dt, device):
    """
    Generate trajectory using semi-implicit Euler with full autodiff support.
    
    Semi-implicit Euler (velocity Verlet) is more stable than explicit Euler
    and matches Gym's integration. Despite not using torchdiffeq's ODE solver,
    gradient flow is maintained through pure PyTorch operations.
    
    This is crucial for accurate pendulum dynamics and for APHYNITY training.
    """
    trajectory = [initial_state]
    current_state = initial_state
    
    for step in range(num_steps):
        # Extract state components
        theta = current_state[:, 0:1]
        omega = current_state[:, 1:2]
        
        # Compute derivatives at current state (using physics prior)
        action = torch.zeros(current_state.shape[0], 1, device=device)
        derivatives = physics_prior(current_state, action)
        dtheta_dt = derivatives[:, 0:1]  
        domega_dt = derivatives[:, 1:2]
        
        # Semi-implicit Euler: velocity step first
        omega_new = omega + domega_dt * dt
        
        # Then position step using updated velocity
        theta_new = theta + omega_new * dt
        
        # Stack and append to trajectory
        current_state = torch.cat([theta_new, omega_new], dim=1)
        trajectory.append(current_state)
    
    return torch.stack(trajectory)


def compute_error(prior_pred, gym_truth):
    prior_pred = prior_pred.squeeze() if prior_pred.dim() > 1 else prior_pred
    gym_truth = gym_truth.squeeze() if gym_truth.dim() > 1 else gym_truth
    
    mse_theta = ((prior_pred[0] - gym_truth[0]) ** 2).item()
    mse_omega = ((prior_pred[1] - gym_truth[1]) ** 2).item()
    rmse_theta = np.sqrt(mse_theta)
    rmse_omega = np.sqrt(mse_omega)
    return mse_theta, mse_omega, rmse_theta, rmse_omega


def state_to_observation(state_np):
    """
    Convert raw state [θ, ω] to observation [cos(θ), sin(θ), ω].
    """
    theta, omega = state_np[0], state_np[1]
    return np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)


def main(friction_alpha=0.0):
    """
    Main test function.
    
    Args:
        friction_alpha: Friction coefficient. 0.0 = no friction (perfect prior), 
                       higher values add damping to test residual learning
    
    Note: Uses RK4 via torchdiffeq for all integration to maintain gradient flow
    for residual learning during APHYNITY training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Friction (alpha): {friction_alpha}")
    print(f"Integration: Semi-Implicit Euler - autodiff enabled")
    print()
    
    env = gym.make("Pendulum-v1")
    
    # Match Gym's actual dynamics: θ'' = 15*sin(θ) + friction term
    # Gym uses: newthdot = thdot + (3*g/(2*l) * sin(th)) * dt
    # With g=10, l=1: coefficient = 3*10/(2*1) = 15
    # Friction is applied as: -alpha * omega
    physics_prior = PendulumPhysics(omega0_square=15.0, alpha=friction_alpha).to(device)
    
    initial_states = [
        torch.tensor([[0.5, 0.0]], device=device, dtype=torch.float32),
        torch.tensor([[1.57, 0.0]], device=device, dtype=torch.float32),
        torch.tensor([[0.0, 2.0]], device=device, dtype=torch.float32),
    ]
    
    dt = 0.05  # Gym Pendulum-v1 dt
    horizons = [1, 2, 3, 5, 10,20,30,40]  # Steps
    max_steps = max(horizons)
    
    print("=" * 100)
    print("PENDULUM PHYSICS PRIOR TEST - SEMI-IMPLICIT EULER")
    print("Approach: APHYNITY-Style Multi-Step ODE Integration (Error Compounding)")
    print("=" * 100)
    
    all_results = {}
    
    for horizon in horizons:
        all_results[horizon] = []
    
    for traj_idx, initial_state in enumerate(initial_states):
        print(f"\nTrajectory {traj_idx + 1}: θ₀={initial_state[0, 0]:.4f}, ω₀={initial_state[0, 1]:.4f}")
        
        # Convert initial state to observation for gym interface
        initial_obs = state_to_observation(initial_state[0].cpu().numpy())
        
        # Generate ground truth from Gym (using observations, decoding internally to states)
        gt_trajectory = generate_gym_trajectory(env, initial_obs, max_steps)
        
        # Generate prediction from our physics prior using semi-implicit Euler with autodiff support
        # (Required for accurate pendulum dynamics and residual learning in APHYNITY training)
        pred_trajectory = generate_prior_trajectory_semiimplicit_euler_autodiff(physics_prior, initial_state, max_steps, dt, device)
        
        for horizon in horizons:
            pred_at_h = pred_trajectory[horizon]
            gt_at_h = gt_trajectory[horizon].to(device)
            
            mse_theta, mse_omega, rmse_theta, rmse_omega = compute_error(pred_at_h, gt_at_h)
            
            rmse_total = np.sqrt(mse_theta + mse_omega)
            
            print(f"  Horizon {horizon:2d} ({horizon*dt:.3f}s): RMSE_θ={rmse_theta:.6f}, RMSE_ω={rmse_omega:.6f}, Total={rmse_total:.6f}")
            
            all_results[horizon].append({
                'traj': traj_idx + 1,
                'rmse_theta': rmse_theta,
                'rmse_omega': rmse_omega,
                'rmse_total': rmse_total
            })
    
    print("\n" + "=" * 100)
    print("SUMMARY - PRIOR ACCURACY AT DIFFERENT HORIZONS")
    print("=" * 100)
    
    for horizon in horizons:
        results = all_results[horizon]
        rmse_totals = [r['rmse_total'] for r in results]
        
        print(f"Horizon {horizon:2d} steps ({horizon*dt:.3f}s):")
        print(f"  Mean RMSE: {np.mean(rmse_totals):.6f}")
        print(f"  Std RMSE:  {np.std(rmse_totals):.6f}")
        print(f"  Max RMSE:  {np.max(rmse_totals):.6f}")


if __name__ == "__main__":
    # Test 1: No friction (perfect prior - should match Gym exactly)
    print("\n" + "="*100)
    print("TEST 1: NO FRICTION (PERFECT PRIOR)")
    print("Expected: Near-perfect match with Gym environment")
    print("="*100 + "\n")
    main(friction_alpha=0.0)
    
    # Test 2: With friction (imperfect prior - simulates real-world physics)
    print("\n\n" + "="*100)
    print("TEST 2: WITH FRICTION (imperfect prior - residual must learn this)")
    print("Expected: Significant errors that grow with horizon")
    print("="*100 + "\n")
    main(friction_alpha=0.2)
