import torch
import numpy as np
import gymnasium as gym
import math
import sys
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')
from pendulum_physics import PendulumPhysics
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from torchdiffeq import odeint


def generate_gym_trajectory(env, initial_state, time_steps):
    trajectory = []
    state_np = initial_state[0].cpu().numpy()
    obs, _ = env.reset(seed=None)
    env.env.state = state_np
    trajectory.append(torch.tensor(env.env.state.copy(), dtype=torch.float32))
    
    for step in range(len(time_steps) - 1):
        action = np.zeros(1)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(torch.tensor(env.env.state.copy(), dtype=torch.float32))
        if terminated or truncated:
            break
    
    return torch.stack(trajectory)


def generate_prior_trajectory_odeint(physics_prior, initial_state, time_steps, device):
    dummy_action = torch.zeros(initial_state.shape[0], 1, device=device)
    
    def ode_dynamics(t, state_t):
        return physics_prior(state_t, dummy_action)
    
    trajectory = odeint(ode_dynamics, initial_state, time_steps, method='dopri8', rtol=1e-8, atol=1e-9)
    return trajectory


def compute_error(prior_pred, gym_truth):
    prior_pred = prior_pred.squeeze() if prior_pred.dim() > 1 else prior_pred
    gym_truth = gym_truth.squeeze() if gym_truth.dim() > 1 else gym_truth
    
    mse_theta = ((prior_pred[0] - gym_truth[0]) ** 2).item()
    mse_omega = ((prior_pred[1] - gym_truth[1]) ** 2).item()
    rmse_theta = np.sqrt(mse_theta)
    rmse_omega = np.sqrt(mse_omega)
    return mse_theta, mse_omega, rmse_theta, rmse_omega


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    env = gym.make("Pendulum-v1")
    
    physics_prior = PendulumPhysics(omega0_square=(2 * math.pi / 6) ** 2, alpha=0.2).to(device)
    
    initial_states = [
        torch.tensor([[0.5, 0.0]], device=device, dtype=torch.float32),
        torch.tensor([[1.57, 0.0]], device=device, dtype=torch.float32),
        torch.tensor([[0.0, 2.0]], device=device, dtype=torch.float32),
    ]
    
    dt = 0.01
    horizons = [1, 3, 5, 10, 20, 30, 40]
    max_steps = max(horizons)
    
    print("=" * 100)
    print("PENDULUM PHYSICS PRIOR ABLATION TEST (APHYNITY-Style ODE Integration)")
    print("Approach: Teacher Forcing via odeint() - Prior Only (No Residuals)")
    print("=" * 100)
    
    all_results = {}
    
    for horizon in horizons:
        all_results[horizon] = []
    
    for traj_idx, initial_state in enumerate(initial_states):
        print(f"\nTrajectory {traj_idx + 1}: θ₀={initial_state[0, 0]:.4f}, ω₀={initial_state[0, 1]:.4f}")
        
        time_steps = torch.arange(0, (max_steps + 1) * dt, dt, dtype=torch.float32, device=device)
        
        gt_trajectory = generate_gym_trajectory(env, initial_state, time_steps)
        
        pred_trajectory = generate_prior_trajectory_odeint(physics_prior, initial_state, time_steps, device)
        
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
    main()
