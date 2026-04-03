#!/usr/bin/env python3
"""
Multi-Step Prediction Test: Physics Prior Only

Tests how prediction error grows over multiple steps when using predicted states
instead of ground truth states. Helps understand model drift/accumulation.

Configuration:
- Trajectory horizon: H = 10 steps
- Number of random trajectories: n = 5
- Only uses physics prior (F_p), no residual network
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from physics_prior import PhysicsPrior

# ====================== CONFIGURATION ======================
TRAJECTORY_HORIZON = 10  # H = 10 steps
NUM_TRAJECTORIES = 5     # n = 5 random trajectory samples
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STATE_DIMS = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
ACTION_DIMS = ['aileron', 'elevator', 'throttle']


def load_sample_transition(csv_path: str = '../data/trajectory_data.csv', sample_idx: int = None):
    """Load a single transition from the CSV dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(df))
    
    row = df.iloc[sample_idx]
    
    state_cols = [f's_t_{i}' for i in range(8)]
    action_cols = [f'a_t_{i}' for i in range(3)]
    next_state_cols = [f's_t+1_{i}' for i in range(8)]
    
    state = torch.tensor([row[col] for col in state_cols], dtype=torch.float32)
    action = torch.tensor([row[col] for col in action_cols], dtype=torch.float32)
    next_state = torch.tensor([row[col] for col in next_state_cols], dtype=torch.float32)
    
    return state, action, next_state


def extract_trajectory_sequence(csv_path: str, start_idx: int, horizon: int):
    """
    Extract a sequence of states and actions from the dataset.
    
    Returns:
        states: list of (8,) tensors
        actions: list of (3,) tensors
        next_states: list of (8,) tensors (ground truth for validation)
    """
    df = pd.read_csv(csv_path)
    
    # Ensure we have enough data
    if start_idx + horizon > len(df):
        return None
    
    states = []
    actions = []
    next_states = []
    
    for i in range(horizon):
        idx = start_idx + i
        row = df.iloc[idx]
        
        state_cols = [f's_t_{j}' for j in range(8)]
        action_cols = [f'a_t_{j}' for j in range(3)]
        next_state_cols = [f's_t+1_{j}' for j in range(8)]
        
        state = torch.tensor([row[col] for col in state_cols], dtype=torch.float32)
        action = torch.tensor([row[col] for col in action_cols], dtype=torch.float32)
        next_state = torch.tensor([row[col] for col in next_state_cols], dtype=torch.float32)
        
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
    
    return states, actions, next_states


def multi_step_rollout(physics_prior, states, actions, next_states_true):
    """
    Perform multi-step rollout using physics prior.
    
    Returns:
        - errors_per_step: L2 errors at each step
        - errors_relative_per_step: Relative errors at each step
        - state_errors_per_dim: Errors for each state dimension at each step
    """
    horizon = len(states)
    errors_per_step = []
    errors_relative_per_step = []
    state_errors_per_dim = []  # (horizon, 8) - error for each dimension at each step
    
    # Start with ground truth first state
    current_state = states[0].unsqueeze(0).to(DEVICE)  # (1, 8)
    
    for step in range(horizon):
        current_action = actions[step].unsqueeze(0).to(DEVICE)  # (1, 3)
        ground_truth_next = next_states_true[step].to(DEVICE)    # (8,)
        
        # Predict next state using physics prior
        with torch.no_grad():
            # Get derivatives from physics prior
            dx_dt = physics_prior(current_state, current_action)  # (1, 8)
            
            # Simple Euler step: s_t+1 = s_t + dt * dx_dt
            # For our 0.01s timestep with 10 RK4 substeps of 0.001s
            predicted_next = current_state + 0.01 * dx_dt
            predicted_next = predicted_next.squeeze(0)  # (8,)
        
        # Calculate error
        error = torch.norm(predicted_next - ground_truth_next).item()
        
        # Calculate relative error
        ground_truth_magnitude = torch.norm(ground_truth_next).item() + 1e-6
        relative_error = error / ground_truth_magnitude
        
        # State-by-state errors
        state_errors = torch.abs(predicted_next - ground_truth_next).cpu().numpy()
        
        errors_per_step.append(error)
        errors_relative_per_step.append(relative_error)
        state_errors_per_dim.append(state_errors)
        
        # Use predicted state for next step (NOT ground truth)
        current_state = predicted_next.unsqueeze(0)
    
    state_errors_per_dim = np.array(state_errors_per_dim)  # (horizon, 8)
    
    return errors_per_step, errors_relative_per_step, state_errors_per_dim


def main():
    print("\n" + "="*80)
    print("MULTI-STEP PREDICTION TEST: Physics Prior Only")
    print("="*80)
    print(f"Trajectory Horizon: H = {TRAJECTORY_HORIZON} steps")
    print(f"Number of Trajectories: n = {NUM_TRAJECTORIES}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")
    
    # Load physics prior
    print("Loading Physics Prior...")
    physics_prior = PhysicsPrior(config_path='aero_coefficients.yaml')
    physics_prior = physics_prior.to(DEVICE)
    physics_prior.eval()
    print("✓ Physics prior loaded\n")
    
    # Load dataset
    csv_path = Path('../data/trajectory_data.csv')
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    max_start_idx = len(df) - TRAJECTORY_HORIZON
    
    print(f"Dataset size: {len(df)} transitions")
    print(f"Max available trajectories of length {TRAJECTORY_HORIZON}: {max_start_idx}\n")
    
    if max_start_idx < NUM_TRAJECTORIES:
        print(f"Warning: Dataset too small. Available: {max_start_idx}, Requested: {NUM_TRAJECTORIES}")
        num_trajectories = min(NUM_TRAJECTORIES, max_start_idx)
    else:
        num_trajectories = NUM_TRAJECTORIES
    
    # Track results across all trajectories
    all_errors = []  # (num_trajectories, horizon)
    all_relative_errors = []
    all_state_errors = []  # (num_trajectories, horizon, 8)
    
    print("="*80)
    print("RUNNING MULTI-STEP PREDICTIONS")
    print("="*80 + "\n")
    
    for traj_idx in range(num_trajectories):
        # Sample random starting index for trajectory
        start_idx = np.random.randint(0, max_start_idx)
        
        print(f"Trajectory {traj_idx + 1}/{num_trajectories}")
        print(f"  Starting from dataset index: {start_idx}")
        
        # Extract trajectory sequence
        result = extract_trajectory_sequence(csv_path, start_idx, TRAJECTORY_HORIZON)
        if result is None:
            print(f"  ✗ Failed to extract trajectory")
            continue
        
        states, actions, next_states_true = result
        
        print(f"  Initial state: phi={states[0][0]:.6f}, theta={states[0][1]:.6f}, Va={states[0][2]:.2f}, p={states[0][3]:.4f}, q={states[0][4]:.4f}, r={states[0][5]:.4f}")
        
        # Run multi-step rollout
        errors, rel_errors, state_errors = multi_step_rollout(
            physics_prior, states, actions, next_states_true
        )
        
        all_errors.append(errors)
        all_relative_errors.append(rel_errors)
        all_state_errors.append(state_errors)
        
        # Print step-by-step results for this trajectory
        print(f"\n  Step-by-step errors:")
        print(f"  Step | L2 Error  | Rel Error |  phi err  | theta err |  Va err   |   p err   |   q err   |   r err   | alpha err | beta err  |")
        print(f"  -----|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|")
        
        for step in range(TRAJECTORY_HORIZON):
            phi_err = state_errors[step][0]
            theta_err = state_errors[step][1]
            va_err = state_errors[step][2]
            p_err = state_errors[step][3]
            q_err = state_errors[step][4]
            r_err = state_errors[step][5]
            alpha_err = state_errors[step][6]
            beta_err = state_errors[step][7]
            
            print(f"   {step:2d}  | {errors[step]:9.6f} | {rel_errors[step]:9.6f} | {phi_err:9.6f} | {theta_err:9.6f} | {va_err:9.6f} | {p_err:9.6f} | {q_err:9.6f} | {r_err:9.6f} | {alpha_err:9.6f} | {beta_err:9.6f} |")
        
        print(f"\n  Summary for this trajectory:")
        print(f"    Max L2 Error: {max(errors):.6f} (at step {np.argmax(errors)})")
        print(f"    Final L2 Error: {errors[-1]:.6f}")
        print(f"    Mean L2 Error: {np.mean(errors):.6f}")
        print(f"    Max Rel Error: {max(rel_errors):.6f}")
        print(f"    Final Rel Error: {rel_errors[-1]:.6f}")
        print()
    
    # ============================ AGGREGATE STATISTICS ============================
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS ACROSS ALL TRAJECTORIES")
    print("="*80 + "\n")
    
    all_errors = np.array(all_errors)  # (num_trajectories, horizon)
    all_relative_errors = np.array(all_relative_errors)
    all_state_errors = np.array(all_state_errors)  # (num_trajectories, horizon, 8)
    
    print("Error Growth Over Time (Overall L2 + Angular Rates):")
    print(f"Step | Mean L2 Err | Std Dev   | Mean Rel Err | Std Dev   |   Mean p   |   Mean q   |   Mean r   |")
    print(f"-----|-------------|-----------|-------------|-----------|-----------|-----------|-----------|")
    
    for step in range(TRAJECTORY_HORIZON):
        mean_l2 = np.mean(all_errors[:, step])
        std_l2 = np.std(all_errors[:, step])
        mean_rel = np.mean(all_relative_errors[:, step])
        std_rel = np.std(all_relative_errors[:, step])
        
        mean_p_err = np.mean(all_state_errors[:, step, 3])
        mean_q_err = np.mean(all_state_errors[:, step, 4])
        mean_r_err = np.mean(all_state_errors[:, step, 5])
        
        print(f" {step:2d}  | {mean_l2:11.6f} | {std_l2:9.6f} | {mean_rel:11.6f} | {std_rel:9.6f} | {mean_p_err:10.6f} | {mean_q_err:10.6f} | {mean_r_err:10.6f} |")
    
    # Detailed per-dimension error at each step
    print("\n" + "="*120)
    print("DETAILED PER-DIMENSION ERROR BREAKDOWN (Mean across trajectories at each step)")
    print("="*120 + "\n")
    
    dim_names = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
    
    for dim in range(8):
        print(f"\n{dim_names[dim].upper()} (dimension {dim}):")
        print(f"Step | Mean Error | Std Dev  | Min Error | Max Error |")
        print(f"-----|------------|----------|-----------|-----------|")
        
        for step in range(TRAJECTORY_HORIZON):
            errors_this_step = all_state_errors[:, step, dim]
            mean_err = np.mean(errors_this_step)
            std_err = np.std(errors_this_step)
            min_err = np.min(errors_this_step)
            max_err = np.max(errors_this_step)
            
            print(f" {step:2d}  | {mean_err:10.6f} | {std_err:8.6f} | {min_err:9.6f} | {max_err:9.6f} |")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total trajectories tested: {num_trajectories}")
    print(f"Trajectory length (H): {TRAJECTORY_HORIZON}")
    print(f"Total predictions made: {num_trajectories * TRAJECTORY_HORIZON}")
    print()
    print(f"Global Mean L2 Error: {np.mean(all_errors):.6f}")
    print(f"Global Std Dev L2: {np.std(all_errors):.6f}")
    print(f"Global Min L2 Error: {np.min(all_errors):.6f}")
    print(f"Global Max L2 Error: {np.max(all_errors):.6f}")
    print()
    print(f"First Step Mean Error: {np.mean(all_errors[:, 0]):.6f}")
    print(f"Last Step Mean Error: {np.mean(all_errors[:, -1]):.6f}")
    print()
    mean_error_growth = np.mean(all_errors[:, -1]) - np.mean(all_errors[:, 0])
    print(f"Error Growth (last - first): {mean_error_growth:.6f}")
    print(f"Error Growth Rate: {mean_error_growth / TRAJECTORY_HORIZON:.6f} per step")
    
    # Per-dimension overall statistics
    print("\n" + "-"*80)
    print("Overall Per-Dimension Error Summary (all steps, all trajectories):")
    print("-"*80)
    print(f"Dimension | Mean Error | Std Dev  | Min Error | Max Error |")
    print(f"-----------|------------|----------|-----------|-----------|")
    
    dim_names = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
    for dim in range(8):
        errors_this_dim = all_state_errors[:, :, dim].flatten()
        mean_err = np.mean(errors_this_dim)
        std_err = np.std(errors_this_dim)
        min_err = np.min(errors_this_dim)
        max_err = np.max(errors_this_dim)
        
        print(f"{dim_names[dim]:10s} | {mean_err:10.6f} | {std_err:8.6f} | {min_err:9.6f} | {max_err:9.6f} |")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
