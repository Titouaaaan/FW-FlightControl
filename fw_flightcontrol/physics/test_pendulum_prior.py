import torch
import numpy as np
import gymnasium as gym
import math
import sys
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')
from pendulum_physics import PendulumPhysics
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel


def generate_gym_trajectory(env, initial_state, num_steps):
    trajectory = []
    state_np = initial_state[0].cpu().numpy()
    obs, _ = env.reset(seed=None)
    env.env.state = state_np
    trajectory.append(torch.tensor(env.env.state.copy(), dtype=torch.float32))
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(torch.tensor(env.env.state.copy(), dtype=torch.float32))
        if terminated or truncated:
            break
    
    return torch.stack(trajectory)


def generate_prior_trajectory(hybrid_model, initial_state, num_steps):
    trajectory = torch.zeros(num_steps + 1, 2, dtype=torch.float32, device=initial_state.device)
    trajectory[0] = initial_state
    state = initial_state.clone()
    
    dummy_action = torch.zeros(state.shape[0], 1, device=initial_state.device)
    
    with torch.no_grad():
        for step in range(num_steps):
            state = hybrid_model.integrate_rk4(state, dummy_action)
            trajectory[step + 1] = state
    
    return trajectory


def compute_error(prior_pred, gym_truth):
    mse_theta = ((prior_pred[:, 0] - gym_truth[:, 0]) ** 2).mean().item()
    mse_omega = ((prior_pred[:, 1] - gym_truth[:, 1]) ** 2).mean().item()
    rmse_theta = np.sqrt(mse_theta)
    rmse_omega = np.sqrt(mse_omega)
    return mse_theta, mse_omega, rmse_theta, rmse_omega


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    env = gym.make("Pendulum-v1")
    
    physics_prior = PendulumPhysics(omega0_square=(2 * math.pi / 6) ** 2, alpha=0.2).to(device)
    
    dummy_residual = PhysicsAugmented(state_dim=2, action_dim=1, hidden_dims=[64], activation='relu').to(device)
    
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=dummy_residual,
        with_prior=True,
        with_residual=False
    ).to(device)
    
    initial_states = [
        torch.tensor([[0.5, 0.0]], device=device),
        torch.tensor([[1.57, 0.0]], device=device),
        torch.tensor([[0.0, 2.0]], device=device),
    ]
    
    horizons = [1, 3, 5, 10,20,30,40]
    
    print("=" * 100)
    print("PENDULUM PHYSICS PRIOR TEST (Gym Environment vs Physics Prior)")
    print("=" * 100)
    
    all_results = {}
    
    for horizon in horizons:
        all_results[horizon] = []
    
    for traj_idx, initial_state in enumerate(initial_states):
        print(f"\nTrajectory {traj_idx + 1}: θ₀={initial_state[0, 0]:.4f}, ω₀={initial_state[0, 1]:.4f}")
        
        gt_trajectory = generate_gym_trajectory(env, initial_state, max(horizons))
        
        for horizon in horizons:
            pred_trajectory = generate_prior_trajectory(hybrid_model, initial_state, horizon)
            
            pred_at_h = pred_trajectory[horizon].to(device)
            gt_at_h = gt_trajectory[horizon].to(device)
            
            mse_theta, mse_omega, rmse_theta, rmse_omega = compute_error(
                pred_at_h.unsqueeze(0), gt_at_h.unsqueeze(0)
            )
            
            rmse_total = np.sqrt(mse_theta + mse_omega)
            
            print(f"  Horizon {horizon:2d}: RMSE_θ={rmse_theta:.6f}, RMSE_ω={rmse_omega:.6f}, Total={rmse_total:.6f}")
            
            all_results[horizon].append({
                'traj': traj_idx + 1,
                'rmse_theta': rmse_theta,
                'rmse_omega': rmse_omega,
                'rmse_total': rmse_total
            })
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    for horizon in horizons:
        results = all_results[horizon]
        rmse_totals = [r['rmse_total'] for r in results]
        
        print(f"Horizon {horizon:2d} steps:")
        print(f"  Mean RMSE: {np.mean(rmse_totals):.6f}")
        print(f"  Std RMSE:  {np.std(rmse_totals):.6f}")
        print(f"  Max RMSE:  {np.max(rmse_totals):.6f}")


if __name__ == "__main__":
    main()
