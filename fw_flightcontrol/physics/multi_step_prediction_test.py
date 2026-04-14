#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from torchdiffeq import odeint

sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')
from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = 0.01
STATE_DIMS = 8
ACTION_DIMS = 3
HORIZONS = [1, 3, 5, 10, 20]
TEST_INDICES = [0, 100, 300]

STATE_COL_NAMES = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
ACTION_COL_NAMES = ['aileron', 'elevator', 'throttle']


def load_trajectory_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def extract_trajectory_sequence(df, start_idx, max_horizon):
    if start_idx + max_horizon >= len(df):
        return None
    
    state_cols = [f's_t_{i}' for i in range(STATE_DIMS)]
    action_cols = [f'a_t_{i}' for i in range(ACTION_DIMS)]
    next_state_cols = [f's_t+1_{i}' for i in range(STATE_DIMS)]
    
    states = []
    actions = []
    next_states = []
    
    for i in range(max_horizon):
        idx = start_idx + i
        row = df.iloc[idx]
        
        state = torch.tensor([row[col] for col in state_cols], dtype=torch.float32)
        action = torch.tensor([row[col] for col in action_cols], dtype=torch.float32)
        next_state = torch.tensor([row[col] for col in next_state_cols], dtype=torch.float32)
        
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
    
    return states, actions, next_states


def generate_ground_truth_trajectory(states_sequence, num_steps):
    trajectory = []
    for i in range(num_steps):
        trajectory.append(states_sequence[i])
    return torch.stack(trajectory)


def generate_prior_prediction_odeint(physics_prior, initial_state, actions, time_steps, device):
    dummy_action = torch.zeros(initial_state.shape[0], ACTION_DIMS, device=device, dtype=torch.float32)
    
    def ode_dynamics(t, state_t):
        return physics_prior(state_t, dummy_action)
    
    trajectory = odeint(ode_dynamics, initial_state, time_steps, 
                       method='dopri8', rtol=1e-8, atol=1e-9)
    
    return trajectory


def compute_error_at_horizon(pred_traj, gt_traj, horizon_idx):
    if horizon_idx >= len(pred_traj) or horizon_idx >= len(gt_traj):
        return None
    
    pred_state = pred_traj[horizon_idx]
    gt_state = gt_traj[horizon_idx]
    
    mse = ((pred_state - gt_state) ** 2).mean().item()
    rmse = np.sqrt(mse)
    mae = torch.abs(pred_state - gt_state).mean().item()
    
    return rmse, mae


def main():
    print("\n" + "="*100)
    print("UAV PHYSICS PRIOR ABLATION TEST (APHYNITY-Style ODE Integration)")
    print("Approach: Teacher Forcing via odeint() - Prior Only (No Residuals)")
    print("="*100 + "\n")
    
    csv_path = Path('/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/data/updated_trajectory_data_noatmo.csv')
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    print(f"Loading trajectory data from {csv_path}...")
    df = load_trajectory_data(csv_path)
    print(f"Dataset size: {len(df)} transitions\n")
    
    print("Initializing Physics Prior...")
    physics_prior = PhysicsPrior(config_path='aero_coefficients.yaml')
    physics_prior = physics_prior.to(DEVICE)
    physics_prior.eval()
    print("✓ Physics prior loaded\n")
    
    max_horizon = max(HORIZONS)
    all_results = {h: [] for h in HORIZONS}
    
    print("="*100)
    print(f"Testing Prior at Indices: {TEST_INDICES}")
    print(f"Horizons: {HORIZONS}")
    print(f"Each horizon step = {DT} seconds")
    print("="*100 + "\n")
    
    for test_idx in TEST_INDICES:
        print(f"Test Index {test_idx}:")
        
        trajectory_data = extract_trajectory_sequence(df, test_idx, max_horizon)
        
        if trajectory_data is None:
            print(f"  ✗ Not enough data from index {test_idx}")
            continue
        
        states_seq, actions_seq, next_states_seq = trajectory_data
        
        initial_state = states_seq[0].to(DEVICE)
        
        time_steps = torch.arange(0, (max_horizon + 1) * DT, DT, dtype=torch.float32, device=DEVICE)
        
        gt_trajectory = generate_ground_truth_trajectory(
            [s.to(DEVICE) for s in states_seq] + [next_states_seq[-1].to(DEVICE)], 
            max_horizon + 1
        )
        
        pred_trajectory = generate_prior_prediction_odeint(
            physics_prior, initial_state.unsqueeze(0), actions_seq, time_steps, DEVICE
        )
        
        for horizon in HORIZONS:
            error_data = compute_error_at_horizon(pred_trajectory, gt_trajectory, horizon)
            
            if error_data is not None:
                rmse, mae = error_data
                print(f"  Horizon {horizon:2d} ({horizon*DT:.3f}s): RMSE={rmse:.6f}, MAE={mae:.6f}")
                all_results[horizon].append({
                    'test_idx': test_idx,
                    'rmse': rmse,
                    'mae': mae
                })
        
        print()
    
    print("="*100)
    print("SUMMARY - PRIOR ACCURACY AT DIFFERENT HORIZONS")
    print("="*100 + "\n")
    
    for horizon in HORIZONS:
        if all_results[horizon]:
            rmses = [r['rmse'] for r in all_results[horizon]]
            maes = [r['mae'] for r in all_results[horizon]]
            
            print(f"Horizon {horizon:2d} steps ({horizon*DT:.3f}s):")
            print(f"  Mean RMSE: {np.mean(rmses):.6f}")
            print(f"  Std RMSE:  {np.std(rmses):.6f}")
            print(f"  Max RMSE:  {np.max(rmses):.6f}")
            print(f"  Mean MAE:  {np.mean(maes):.6f}")
        else:
            print(f"Horizon {horizon:2d}: No data")
        print()


if __name__ == '__main__':
    main()
