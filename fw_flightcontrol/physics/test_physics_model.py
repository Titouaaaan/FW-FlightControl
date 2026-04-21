#!/usr/bin/env python3
"""
Test script for full physics-augmented model (APHYNITY).

Tests the combined dynamics: ds/dt = F_p(s,u) + F_a(s,u)
where F_p is the frozen physics prior and F_a is the learned residual network.

Strategy:
- Load random trajectories of horizon length 10
- For each trajectory, use 3 random starting points from early/medium/late intervals
- Evaluate prediction accuracy at 1, 3, 5, and 10 steps
- Report mean MAE and standard deviation for each horizon
- Generate visualizations comparing predictions vs ground truth

"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')
from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel

# Configuration
DEVICE = torch.device('cpu')
DT = 0.01
STATE_DIMS = 8
ACTION_DIMS = 3
MAX_HORIZON = 10
HORIZONS = [1, 3, 5, 10]
FIG_DPI = 300

STATE_COL_NAMES = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
STATE_NAMES = ['φ (roll)', 'θ (pitch)', 'V_a (speed)', 'p (roll rate)', 
               'q (pitch rate)', 'r (yaw rate)', 'α (AoA)', 'β (sideslip)']


def load_trajectory_data(csv_path):
    """Load trajectory data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def get_random_test_sequences(df, num_trajectories=10, num_points_per_traj=3):
    """
    Extract random test sequences from trajectories using stratified sampling.
    
    For each trajectory, sample 3 starting points from:
    - Early interval:  step_id ~6
    - Medium interval: step_id ~100
    - Late interval:   step_id ~300
    
    Args:
        df: DataFrame with trajectory data
        num_trajectories: Number of trajectories to sample (0 to num_trajectories-1)
        num_points_per_traj: Number of starting points per trajectory (typically 3)
    
    Returns:
        List of (trajectory_id, step_id, start_row_idx) tuples
    """
    test_sequences = []
    
    # Define intervals with some randomness
    interval_points = [
        (0, 20, "early"),      # Early: steps 0-20, target ~6
        (80, 120, "medium"),   # Medium: steps 80-120, target ~100
        (280, 320, "late")     # Late: steps 280-320, target ~300
    ]
    
    for traj_id in range(num_trajectories):
        traj_df = df[df['trajectory_id'] == traj_id]
        
        if len(traj_df) < MAX_HORIZON + 20:  # Need enough data
            continue
        
        traj_start_rows = traj_df.index.tolist()
        
        for min_step, max_step, interval_name in interval_points:
            # Sample a random step in this interval
            available_steps = traj_df[
                (traj_df['step_id'] >= min_step) & 
                (traj_df['step_id'] <= max_step)
            ]
            
            if len(available_steps) > 0:
                # Pick one random row from this interval
                chosen_row_idx = available_steps.sample(n=1).index[0]
                chosen_step_id = df.loc[chosen_row_idx, 'step_id']
                test_sequences.append((traj_id, chosen_step_id, chosen_row_idx))
    
    return test_sequences


def extract_trajectory_sequence(df, start_row_idx, max_horizon):
    """
    Extract a trajectory sequence of max_horizon steps starting from start_row_idx.
    
    State indices: [0-5,8-9] (skip [6-7] which are control errors, not aerodynamic angles)
    """
    if start_row_idx + max_horizon >= len(df):
        return None
    
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
    """Create ground truth trajectory from states."""
    trajectory = []
    for i in range(num_steps):
        trajectory.append(states_sequence[i])
    return torch.stack(trajectory)


def generate_hybrid_prediction_odeint(hybrid_model, initial_state, actions, time_steps, device):
    """
    Generate trajectory using ODE integration with the hybrid model.
    
    Args:
        hybrid_model: HybridDynamicsModel instance
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
            return hybrid_model(state_t, action_t)
        
        # Use RK4 for integration
        step_trajectory = odeint(ode_dynamics, current_state, t_span, method='rk4')
        
        # step_trajectory is (2, batch_size, STATE_DIMS), take final state
        current_state = step_trajectory[-1]
        trajectory.append(current_state)
    
    return torch.stack(trajectory)


def compute_error_at_horizon(pred_traj, gt_traj, horizon_idx):
    """
    Compute RMSE and MAE at a specific horizon index.
    
    Args:
        pred_traj: Predicted trajectory
        gt_traj: Ground truth trajectory
        horizon_idx: Index of the horizon to evaluate
    
    Returns:
        (overall_rmse, overall_mae, per_state_rmse, per_state_mae, gt_state_values)
    """
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
    
    per_state_rmse = per_state_sq_error.cpu().numpy()
    per_state_mae = per_state_abs_error.cpu().numpy()
    
    # Ground truth values
    gt_values = gt_state.cpu().numpy()
    
    # Overall errors
    overall_mse = per_state_sq_error.mean().item()
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = per_state_abs_error.mean().item()
    
    return overall_rmse, overall_mae, per_state_rmse, per_state_mae, gt_values


def load_hybrid_model(physics_prior_path, residual_checkpoint_path, device):
    """
    Load the hybrid model (physics prior + trained residual network).
    
    Args:
        physics_prior_path: Path to aero_coefficients.yaml
        residual_checkpoint_path: Path to epoch_800.pt checkpoint
        device: Device to load model to
    
    Returns:
        HybridDynamicsModel instance
    """
    # Load physics prior (frozen)
    print("Loading physics prior...")
    physics_prior = PhysicsPrior(config_path=physics_prior_path)
    physics_prior = physics_prior.to(device)
    physics_prior.eval()
    
    # Create residual network
    print("Initializing residual network...")
    residual_network = PhysicsAugmented(
        state_dim=STATE_DIMS,
        action_dim=ACTION_DIMS,
        hidden_dims=[128, 128],
        activation='relu',
        use_batch_norm=False
    )
    
    # Load trained residual network weights
    print(f"Loading residual network checkpoint from {residual_checkpoint_path}...")
    checkpoint = torch.load(residual_checkpoint_path, map_location=device)
    residual_network.load_state_dict(checkpoint['residual_state'])
    residual_network = residual_network.to(device)
    residual_network.eval()
    
    # Create hybrid model
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True,
        integration_method='rk4'
    )
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()
    
    # Freeze all parameters for inference only
    for param in hybrid_model.parameters():
        param.requires_grad = False
    
    print("✓ Hybrid model loaded and frozen for inference\n")
    
    return hybrid_model


def plot_trajectory_predictions(all_results, plots_dir):
    """
    Create visualization comparing predicted vs ground truth trajectories.
    Shows all 8 state dimensions for the first test sequence at full horizon.
    """
    if not all_results[MAX_HORIZON]:
        print("⚠ No results available for trajectory plotting")
        return
    
    # Get the first result which has the full trajectory info
    first_result = all_results[MAX_HORIZON][0]
    
    # Create figure with 8 subplots (one per state)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle('APHYNITY Hybrid Model: Predicted vs Ground Truth Trajectories (Full 10-Step Horizon)', 
                 fontsize=14, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    # We'll use per_state_mae and gt_values from the first result to visualize
    # Note: These are aggregated across all samples, so we'll show average behavior
    
    for state_idx in range(STATE_DIMS):
        ax = axes_flat[state_idx]
        ax.set_title(f'{STATE_NAMES[state_idx]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Horizon Step')
        ax.set_ylabel('State Value')
        ax.grid(True, alpha=0.3)
    
    # Add a note about data aggregation
    fig.text(0.5, 0.02, 'Note: Visualization shows example trajectory. Full statistics reported in terminal output.', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    save_path = plots_dir / 'aphynity_trajectory_comparison.png'
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved trajectory comparison plot: {save_path.name}")


def plot_mae_vs_horizon(all_results, plots_dir):
    """
    Create plot showing how relative error grows with prediction horizon for each state.
    Uses percentage error (MAE / abs(ground_truth)) for fair comparison across scales.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    state_relative_errors_by_horizon = {h: [] for h in HORIZONS}
    
    # Collect per-state relative error for each horizon
    for horizon in HORIZONS:
        if all_results[horizon]:
            per_state_maes = np.array([r['per_state_mae'] for r in all_results[horizon]])
            gt_values_list = np.array([r['gt_values'] for r in all_results[horizon]])
            
            # Compute relative error: (MAE / |mean(ground_truth)|) * 100
            mean_gt_per_state = np.mean(np.abs(gt_values_list), axis=0)  # (8,)
            mean_mae_per_state = np.mean(per_state_maes, axis=0)  # (8,)
            
            # Avoid division by zero
            relative_error = np.zeros(STATE_DIMS)
            for i in range(STATE_DIMS):
                if mean_gt_per_state[i] > 1e-6:
                    relative_error[i] = (mean_mae_per_state[i] / mean_gt_per_state[i]) * 100
                else:
                    relative_error[i] = np.inf  # Undefined for near-zero values
            
            state_relative_errors_by_horizon[horizon] = relative_error
    
    # Plot one line per state
    colors = plt.cm.tab10(np.linspace(0, 1, STATE_DIMS))
    
    for state_idx in range(STATE_DIMS):
        errors = [state_relative_errors_by_horizon[h][state_idx] if h in state_relative_errors_by_horizon and len(state_relative_errors_by_horizon[h]) > 0 else None 
                for h in HORIZONS]
        
        # Filter out None and inf values
        valid_horizons = [HORIZONS[i] for i in range(len(HORIZONS)) if errors[i] is not None and errors[i] != np.inf]
        valid_errors = [e for e in errors if e is not None and e != np.inf]
        
        if valid_errors:
            ax.plot(valid_horizons, valid_errors, marker='o', linewidth=2.5, markersize=8,
                   label=STATE_NAMES[state_idx], color=colors[state_idx])
    
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('APHYNITY: Prediction Error Growth Over Horizon (Relative %)', fontsize=13, fontweight='bold')
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    save_path = plots_dir / 'aphynity_relative_error_vs_horizon.png'
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved relative error vs horizon plot: {save_path.name}")


def plot_error_heatmap(all_results, plots_dir):
    """
    Create heatmap showing relative error (%) for each state at each horizon.
    Uses percentage error (MAE / |ground_truth|) for fair comparison across scales.
    """
    # Build matrix: rows=states, cols=horizons
    relative_error_matrix = np.zeros((STATE_DIMS, len(HORIZONS)))
    
    for h_idx, horizon in enumerate(HORIZONS):
        if all_results[horizon]:
            per_state_maes = np.array([r['per_state_mae'] for r in all_results[horizon]])
            gt_values_list = np.array([r['gt_values'] for r in all_results[horizon]])
            
            # Compute relative error
            mean_gt_per_state = np.mean(np.abs(gt_values_list), axis=0)  # (8,)
            mean_mae_per_state = np.mean(per_state_maes, axis=0)  # (8,)
            
            for i in range(STATE_DIMS):
                if mean_gt_per_state[i] > 1e-6:
                    relative_error_matrix[i, h_idx] = (mean_mae_per_state[i] / mean_gt_per_state[i]) * 100
                else:
                    relative_error_matrix[i, h_idx] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(relative_error_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_yticks(np.arange(STATE_DIMS))
    ax.set_xticklabels([f'H={h}' for h in HORIZONS])
    ax.set_yticklabels(STATE_NAMES)
    
    # Add text annotations with % symbol
    for i in range(STATE_DIMS):
        for j in range(len(HORIZONS)):
            if not np.isnan(relative_error_matrix[i, j]):
                text = ax.text(j, i, f'{relative_error_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('APHYNITY: Prediction Error Heatmap (Relative % Error)\nStates × Horizons', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
    ax.set_ylabel('State Variables', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Error (%)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = plots_dir / 'aphynity_relative_error_heatmap.png'
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved relative error heatmap plot: {save_path.name}")


def main():
    print("\n" + "="*100)
    print("APHYNITY HYBRID MODEL TEST (Physics Prior + Learned Residual Network)")
    print("Approach: Multi-step prediction with ODE integration")
    print("="*100 + "\n")
    
    # Paths
    csv_path = Path('/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/data/updated_trajectory_data_noatmo.csv')
    physics_prior_path = '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics/aero_coefficients.yaml'
    residual_checkpoint_path = '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics/checkpoints/exp_v0.2/epoch_800.pt'
    
    # Validate paths
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    if not Path(residual_checkpoint_path).exists():
        print(f"Error: {residual_checkpoint_path} not found!")
        return
    
    print(f"Loading trajectory data from {csv_path}...")
    df = load_trajectory_data(csv_path)
    print(f"Dataset size: {len(df)} transitions\n")
    
    # Load hybrid model
    hybrid_model = load_hybrid_model(physics_prior_path, residual_checkpoint_path, DEVICE)
    
    # Generate random test sequences
    test_sequences = get_random_test_sequences(df, num_trajectories=10, num_points_per_traj=3)
    
    print("="*100)
    print(f"Testing Hybrid Model on Random Trajectories")
    print(f"Total test sequences: {len(test_sequences)}")
    print(f"Max horizon: {MAX_HORIZON} steps")
    print(f"Horizons to evaluate: {HORIZONS}")
    print(f"Each step = {DT} seconds")
    print("="*100 + "\n")
    
    # Collect results organized by horizon
    all_results = {h: [] for h in HORIZONS}
    
    for test_idx, (traj_id, step_id, row_idx) in enumerate(test_sequences, 1):
        trajectory_data = extract_trajectory_sequence(df, row_idx, MAX_HORIZON)
        
        if trajectory_data is None:
            print(f"✗ Sequence {test_idx}: Traj {traj_id}, step {step_id} - not enough data")
            continue
        
        states_seq, actions_seq, next_states_seq = trajectory_data
        
        initial_state = states_seq[0].to(DEVICE)
        
        # Generate time points for integration
        time_steps = torch.arange(0, (MAX_HORIZON + 1) * DT, DT, dtype=torch.float32, device=DEVICE)
        
        # Ground truth trajectory (initial state + predicted next states)
        gt_trajectory = generate_ground_truth_trajectory(
            [s.to(DEVICE) for s in states_seq] + [next_states_seq[-1].to(DEVICE)],
            MAX_HORIZON + 1
        )
        
        # Generate hybrid model predictions
        with torch.no_grad():
            pred_trajectory = generate_hybrid_prediction_odeint(
                hybrid_model, initial_state.unsqueeze(0), actions_seq, time_steps, DEVICE
            )
        
        # Evaluate at different horizons
        print(f"[{test_idx:2d}] Trajectory {traj_id:2d}, step_id {step_id:3d}:")
        
        for horizon in HORIZONS:
            error_data = compute_error_at_horizon(pred_trajectory, gt_trajectory, horizon)
            
            if error_data is not None:
                overall_rmse, overall_mae, per_state_rmse, per_state_mae, gt_values = error_data
                print(f"        H={horizon:2d} ({horizon*DT:.3f}s): RMSE={overall_rmse:.6f}, MAE={overall_mae:.6f}")
                
                all_results[horizon].append({
                    'traj_id': traj_id,
                    'step_id': step_id,
                    'rmse': overall_rmse,
                    'mae': overall_mae,
                    'per_state_rmse': per_state_rmse,
                    'per_state_mae': per_state_mae,
                    'gt_values': gt_values
                })
        
        print()
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY - HYBRID MODEL ACCURACY AT DIFFERENT HORIZONS")
    print("="*100 + "\n")
    
    print(f"{'Horizon':<12} {'Steps':<10} {'Time':<10} {'Mean MAE':<15} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
    print("-" * 100)
    
    summary_data = {}
    for horizon in HORIZONS:
        if all_results[horizon]:
            maes = np.array([r['mae'] for r in all_results[horizon]])
            mean_mae = np.mean(maes)
            std_mae = np.std(maes)
            min_mae = np.min(maes)
            max_mae = np.max(maes)
            
            summary_data[horizon] = {
                'mean': mean_mae,
                'std': std_mae,
                'min': min_mae,
                'max': max_mae,
                'count': len(maes)
            }
            
            print(f"{f'H={horizon}':<12} {horizon:<10} {horizon*DT:<10.3f}s {mean_mae:<15.6f} {std_mae:<15.6f} {min_mae:<12.6f} {max_mae:<12.6f}")
        else:
            print(f"{f'H={horizon}':<12} {horizon:<10} {horizon*DT:<10.3f}s {'N/A':<15} {'N/A':<15} {'N/A':<12} {'N/A':<12}")
    
    # Per-state error analysis
    print("\n" + "="*100)
    print("PER-STATE ERROR ANALYSIS (Absolute MAE + Relative % Error)")
    print("="*100 + "\n")
    
    for horizon in HORIZONS:
        if all_results[horizon]:
            print(f"Horizon {horizon} ({horizon*DT:.3f}s):")
            print(f"{'':30} {'Typical Magnitude':<20} {'MAE (Abs)':<20} {'Error (%)':<15}")
            print("-" * 100)
            
            per_state_maes = np.array([r['per_state_mae'] for r in all_results[horizon]])
            gt_values_list = np.array([r['gt_values'] for r in all_results[horizon]])
            
            if len(per_state_maes.shape) == 2 and per_state_maes.shape[1] == STATE_DIMS:
                for state_idx, state_name in enumerate(STATE_NAMES):
                    state_maes = per_state_maes[:, state_idx]
                    state_gt_values = gt_values_list[:, state_idx]
                    
                    mean_mae = np.mean(state_maes)
                    std_mae = np.std(state_maes)
                    
                    # Use mean of absolute values for typical magnitude (not mean which could be near zero)
                    typical_magnitude = np.mean(np.abs(state_gt_values))
                    typical_magnitude_std = np.std(np.abs(state_gt_values))
                    
                    # Compute relative error properly
                    if typical_magnitude > 1e-6:
                        relative_error = (mean_mae / typical_magnitude) * 100
                    else:
                        relative_error = np.inf
                    
                    mag_str = f"{typical_magnitude:.6f} (±{typical_magnitude_std:.6f})"
                    mae_str = f"{mean_mae:.6f} (±{std_mae:.6f})"
                    error_str = f"{relative_error:.2f}%" if relative_error != np.inf else "N/A"
                    
                    print(f"  [{state_idx}] {state_name:20s} {mag_str:<20} {mae_str:<20} {error_str:<15}")
            print()
    
    # Generate visualization plots
    print("\n" + "="*100)
    print("GENERATING VISUALIZATIONS")
    print("="*100 + "\n")
    
    plots_dir = Path('/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics/plots/uav_env')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_mae_vs_horizon(all_results, plots_dir)
    plot_error_heatmap(all_results, plots_dir)
    
    print(f"\n✓ All plots saved to: {plots_dir}\n")
    
    print("="*100)
    print("Test complete!")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
