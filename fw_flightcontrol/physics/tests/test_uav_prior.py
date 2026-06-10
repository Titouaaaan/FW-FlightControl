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

# Force CPU for this test to avoid device mismatch issues
DEVICE = torch.device('cpu')
DT = 0.01
STATE_DIMS = 8
ACTION_DIMS = 3
HORIZONS = [1, 3, 5, 10, 20]

STATE_COL_NAMES = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
ACTION_COL_NAMES = ['aileron', 'elevator', 'throttle']


def load_trajectory_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def get_test_sequences(df, traj_id_max=9, step_ids=[0, 100, 300]):
    """
    Extract test sequences for all trajectory IDs up to traj_id_max,
    at specified step_ids within each trajectory.
    
    Returns: List of (trajectory_id, step_id, start_row_idx) tuples
    """
    test_sequences = []
    
    for traj_id in range(traj_id_max + 1):
        # Filter rows for this trajectory
        traj_df = df[df['trajectory_id'] == traj_id]
        
        if len(traj_df) == 0:
            continue
        
        # Get the starting indices in the full dataframe for this trajectory
        traj_start_rows = traj_df.index.tolist()
        
        for step_id in step_ids:
            # Check if this step_id exists for this trajectory
            step_rows = traj_df[traj_df['step_id'] == step_id]
            
            if len(step_rows) > 0:
                # Get the actual row index in the full dataframe
                row_idx = step_rows.index[0]
                test_sequences.append((traj_id, step_id, row_idx))
    
    return test_sequences


def extract_trajectory_sequence(df, start_row_idx, max_horizon):
    if start_row_idx + max_horizon >= len(df):
        return None
    
    # State indices: [0-5] = [roll, pitch, Va, p, q, r], [8-9] = [alpha, beta]
    # Skip [6-7] which are control errors (roll_err, pitch_err), not aerodynamic angles
    state_indices = [0, 1, 2, 3, 4, 5, 8, 9]
    state_cols = [f's_t_{i}' for i in state_indices]
    action_cols = [f'a_t_{i}' for i in range(ACTION_DIMS)]
    next_state_cols = [f's_t+1_{i}' for i in state_indices]
    
    states = []
    actions = []
    next_states = []
    
    for i in range(max_horizon):
        idx = start_row_idx + i
        row = df.iloc[idx]
        
        # Extract state with correct indices and apply unit conversions
        state_vals = [row[col] for col in state_cols]
        # Convert airspeed from km/h to m/s (index 2 in the 8-state vector)
        state_vals[2] = state_vals[2] / 3.6
        state = torch.tensor(state_vals, dtype=torch.float32)
        
        # Extract actions (already normalized [-1, 1] from PID controller)
        action = torch.tensor([row[col] for col in action_cols], dtype=torch.float32)
        
        # Extract next state with same corrections
        next_state_vals = [row[col] for col in next_state_cols]
        next_state_vals[2] = next_state_vals[2] / 3.6
        next_state = torch.tensor(next_state_vals, dtype=torch.float32)
        
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
    """
    Generate trajectory using ODE integration with actual actions at each step.
    
    Args:
        physics_prior: The physics model
        initial_state: Starting state (batch_size, STATE_DIMS)
        actions: List of action vectors, one per time step
        time_steps: Time points for integration
        device: Device to use
    
    Returns:
        Trajectory: (time_steps, batch_size, STATE_DIMS)
    """
    trajectory = [initial_state]
    current_state = initial_state.clone()
    
    # Integrate step by step with actual actions
    for i in range(len(actions)):
        # Current action for this step (expand to batch dimension if needed)
        action_t = actions[i].unsqueeze(0) if actions[i].dim() == 1 else actions[i]
        
        # Time interval for this step
        t_start = time_steps[i]
        t_end = time_steps[i + 1]
        t_span = torch.stack([t_start, t_end])
        
        # Integrate with constant action
        def ode_dynamics(t, state_t):
            return physics_prior(state_t, action_t)
        
        # Use RK4 for faster integration while maintaining reasonable accuracy
        step_trajectory = odeint(ode_dynamics, current_state, t_span, 
                                method='rk4')
        
        # step_trajectory is (2, batch_size, STATE_DIMS), take final state
        current_state = step_trajectory[-1]
        trajectory.append(current_state)
    
    return torch.stack(trajectory)


def compute_error_at_horizon(pred_traj, gt_traj, horizon_idx):
    if horizon_idx >= len(pred_traj) or horizon_idx >= len(gt_traj):
        return None
    
    pred_state = pred_traj[horizon_idx]  # shape: [batch_size, 8] or [8]
    gt_state = gt_traj[horizon_idx]      # shape: [batch_size, 8] or [8]
    
    # Squeeze batch dimension if present
    if pred_state.dim() > 1:
        pred_state = pred_state.squeeze(0)
        gt_state = gt_state.squeeze(0)
    
    # Per-state errors
    per_state_sq_error = (pred_state - gt_state) ** 2
    per_state_abs_error = torch.abs(pred_state - gt_state)
    
    # Return squared errors (not sqrt!) so RMSE can be computed properly via sqrt(mean(squared_errors))
    per_state_rmse = per_state_sq_error.cpu().numpy()  # Squared error for each state
    per_state_mae = per_state_abs_error.cpu().numpy()  # Absolute error for each state
    
    # Overall errors
    overall_mse = per_state_sq_error.mean().item()
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = per_state_abs_error.mean().item()
    
    return overall_rmse, overall_mae, per_state_rmse, per_state_mae


def main():
    print("\n" + "="*100)
    print("UAV PHYSICS PRIOR ABLATION TEST (APHYNITY-Style ODE Integration)")
    print("Approach: Teacher Forcing via odeint() - Prior Only (No Residuals)")
    print("="*100 + "\n")
    
    csv_path = Path(__file__).parent.parent / 'data' / 'updated_trajectory_data_noatmo.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    print(f"Loading trajectory data from {csv_path}...")
    df = load_trajectory_data(csv_path)
    print(f"Dataset size: {len(df)} transitions\n")
    
    print("Initializing Physics Prior...")
    config_path = '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics/aero_coefficients.yaml'
    physics_prior = PhysicsPrior(config_path=config_path)
    physics_prior = physics_prior.to(DEVICE)
    physics_prior.eval()
    print("✓ Physics prior loaded\n")
    
    max_horizon = max(HORIZONS)
    all_results = {h: [] for h in HORIZONS}
    
    # Get test sequences for trajectories 0-9, at step_ids 0, 100, 300
    test_sequences = get_test_sequences(df, traj_id_max=9, step_ids=[0, 100, 300])
    
    print("="*100)
    print(f"Testing Prior on Trajectories ID 0-9 at step_ids [0, 100, 300]")
    print(f"Total test sequences: {len(test_sequences)}")
    print(f"Horizons: {HORIZONS}")
    print(f"Each horizon step = {DT} seconds")
    print("="*100 + "\n")
    
    # Collect all step_ids and results organized by (step_id, horizon)
    results_by_step_horizon = {}  # (step_id, horizon) -> list of RMSE values
    step_ids_tested = set()
    
    for traj_id, step_id, row_idx in test_sequences:
        print(f"Trajectory {traj_id:2d}, step_id {step_id:3d}:")
        
        trajectory_data = extract_trajectory_sequence(df, row_idx, max_horizon)
        
        if trajectory_data is None:
            print(f"  ✗ Not enough data from row {row_idx} (need {max_horizon} steps)")
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
                overall_rmse, overall_mae, per_state_rmse, per_state_mae = error_data
                print(f"  Horizon {horizon:2d} ({horizon*DT:.3f}s): RMSE={overall_rmse:.6f}, MAE={overall_mae:.6f}")
                
                # Track by trajectory ID and horizon (kept for backward compatibility)
                all_results[horizon].append({
                    'traj_id': traj_id,
                    'step_id': step_id,
                    'rmse': overall_rmse,
                    'mae': overall_mae,
                    'per_state_rmse': per_state_rmse,
                    'per_state_mae': per_state_mae
                })
                
                # Track by (step_id, horizon) for new summary table
                key = (step_id, horizon)
                if key not in results_by_step_horizon:
                    results_by_step_horizon[key] = []
                results_by_step_horizon[key].append(overall_rmse)
                step_ids_tested.add(step_id)
        
        print()
    
    print("="*100)
    print("SUMMARY - PRIOR ACCURACY BY STEP_ID AND HORIZON")
    print("="*100 + "\n")
    
    print("Format: Mean RMSE (±Std Dev)\n")
    
    # Build table
    step_ids_sorted = sorted(list(step_ids_tested))
    horizons_sorted = sorted(HORIZONS)
    
    # Create header
    header = "step_id    " + "  ".join([f"H={h:2d}" for h in horizons_sorted])
    print(header)
    print("-" * len(header))
    
    # Create rows
    for step_id in step_ids_sorted:
        row = f"{step_id:6d}     "
        row_values = []
        for horizon in horizons_sorted:
            key = (step_id, horizon)
            if key in results_by_step_horizon and len(results_by_step_horizon[key]) > 0:
                rmses = np.array(results_by_step_horizon[key])
                mean_rmse = np.mean(rmses)
                std_rmse = np.std(rmses)
                row_values.append(f"{mean_rmse:.4f}(±{std_rmse:.4f})")
            else:
                row_values.append("    N/A       ")
        row += "  ".join(row_values)
        print(row)
    
    print("\n" + "="*100)
    print("DETAILED SUMMARY - PRIOR ACCURACY AT DIFFERENT HORIZONS")
    print("="*100 + "\n")
    
    state_names = ['φ (roll)', 'θ (pitch)', 'V_a (speed)', 'p (roll rate)', 
                   'q (pitch rate)', 'r (yaw rate)', 'α (AoA)', 'β (sideslip)']
    
    for horizon in HORIZONS:
        if all_results[horizon]:
            rmses = np.array([r['rmse'] for r in all_results[horizon]])
            maes = np.array([r['mae'] for r in all_results[horizon]])
            
            # Collect per-state errors across all test indices
            per_state_rmses = np.array([r['per_state_rmse'] for r in all_results[horizon]])
            per_state_maes = np.array([r['per_state_mae'] for r in all_results[horizon]])
            
            print(f"Horizon {horizon:2d} steps ({horizon*DT:.3f}s):")
            print(f"  Overall RMSE: Mean={np.mean(rmses):.6f}, Std={np.std(rmses):.6f}, Max={np.max(rmses):.6f}")
            print(f"  Overall MAE:  Mean={np.mean(maes):.6f}")
            print(f"\n  Per-State Errors (RMSE):")
            
            # Compute mean and std across all test indices for each state variable
            if len(per_state_rmses.shape) == 2 and per_state_rmses.shape[1] == 8:
                for state_idx, state_name in enumerate(state_names):
                    state_abs_errors = per_state_maes[:, state_idx]  # Absolute errors accumulated
                    state_mae = np.mean(state_abs_errors)  # Proper MAE: mean(absolute_errors)
                    state_mae_std = np.std(state_abs_errors)  # Std of absolute errors
                    print(f"    [{state_idx}] {state_name:20s}: MAE={state_mae:.6f} (±{state_mae_std:.6f})")
            else:
                print(f"  WARNING: Unexpected shape for per_state errors: {per_state_rmses.shape}")
        else:
            print(f"Horizon {horizon:2d}: No data")
        print()


if __name__ == '__main__':
    main()
