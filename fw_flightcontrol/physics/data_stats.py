#!/usr/bin/env python3
"""
Analyze trajectory dataset statistics before training dynamics priors.
Measures reward, PID controller accuracy, action spreads, state ranges, and dynamics properties.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# State variable names (ordered as they appear in the environment)
STATE_NAMES = [
    'roll',           # 0: roll angle (rad)
    'pitch',          # 1: pitch angle (rad)
    'yaw',            # 2: yaw angle (rad)
    'p',              # 3: roll rate (rad/s)
    'q',              # 4: pitch rate (rad/s)
    'r',              # 5: yaw rate (rad/s)
    'u',              # 6: forward velocity (m/s)
    'v',              # 7: lateral velocity (m/s)
    'w',              # 8: vertical velocity (m/s)
    'roll_error',     # 9: target_roll - roll (rad)
    'pitch_error',    # 10: target_pitch - pitch (rad)
    'alpha',          # 11: angle of attack (rad)
    'beta',           # 12: sideslip angle (rad)
    'airspeed'        # 13: true airspeed (m/s)
]

ACTION_NAMES = ['aileron', 'elevator', 'throttle']


def load_trajectory_data(csv_path='../data/trajectory_data.csv'):
    """Load trajectory data from CSV."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} transitions")
    return df


def parse_state_columns(df):
    """
    Parse state, action, and next_state from CSV.
    These are stored as flattened columns in the CSV format.
    Format: s_t_0..s_t_13, a_t_0..a_t_2 (aileron, elevator, throttle), s_t+1_0..s_t+1_13
    """
    # Extract state columns s_t_0 to s_t_13
    state_cols = [f's_t_{i}' for i in range(14)]
    action_cols = [f'a_t_{i}' for i in range(3)]
    next_state_cols = [f's_t+1_{i}' for i in range(14)]
    
    # Extract as numpy arrays
    state = df[state_cols].values
    action = df[action_cols].values
    next_state = df[next_state_cols].values
    
    print(f"\n✓ Parsed columns:")
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Next state shape: {next_state.shape}")
    
    return state, action, next_state


def calculate_reward_stats(df):
    """Calculate reward statistics."""
    print("\n" + "="*80)
    print("REWARD STATISTICS")
    print("="*80)
    
    rewards = df['reward'].values
    print(f"  Mean reward:           {np.mean(rewards):10.6f}")
    print(f"  Median reward:         {np.median(rewards):10.6f}")
    print(f"  Std dev:               {np.std(rewards):10.6f}")
    print(f"  Min reward:            {np.min(rewards):10.6f}")
    print(f"  Max reward:            {np.max(rewards):10.6f}")
    print(f"  25th percentile:       {np.percentile(rewards, 25):10.6f}")
    print(f"  75th percentile:       {np.percentile(rewards, 75):10.6f}")
    
    # Reward per trajectory
    traj_rewards = df.groupby('trajectory_id')['reward'].agg(['mean', 'sum', 'std'])
    print(f"\n  Per-trajectory statistics:")
    print(f"    Mean of traj means:  {traj_rewards['mean'].mean():10.6f}")
    print(f"    Sum per trajectory:  {traj_rewards['sum'].mean():10.6f} (avg) ± {traj_rewards['sum'].std():10.6f}")
    
    return rewards


def calculate_action_stats(action):
    """Calculate action statistics."""
    print("\n" + "="*80)
    print("ACTION STATISTICS")
    print("="*80)
    
    for i, name in enumerate(ACTION_NAMES):
        actions = action[:, i]
        print(f"\n  {name.upper()}:")
        print(f"    Mean:                {np.mean(actions):10.6f}")
        print(f"    Std dev:             {np.std(actions):10.6f}")
        print(f"    Min:                 {np.min(actions):10.6f}")
        print(f"    Max:                 {np.max(actions):10.6f}")
        print(f"    Range (max-min):     {np.ptp(actions):10.6f}")
        print(f"    25th percentile:     {np.percentile(actions, 25):10.6f}")
        print(f"    Median:              {np.percentile(actions, 50):10.6f}")
        print(f"    75th percentile:     {np.percentile(actions, 75):10.6f}")
        
        # Count saturation - different ranges for throttle vs control surfaces
        if name == 'throttle':
            # Throttle range is [0, 1]
            at_min = np.sum(np.isclose(actions, 0.0)) / len(actions) * 100
            at_max = np.sum(np.isclose(actions, 1.0)) / len(actions) * 100
            print(f"    Saturation at 0:     {at_min:.2f}%")
            print(f"    Saturation at 1:     {at_max:.2f}%")
        else:
            # Aileron/elevator range is [-1, 1]
            at_minus_one = np.sum(np.isclose(actions, -1.0)) / len(actions) * 100
            at_plus_one = np.sum(np.isclose(actions, 1.0)) / len(actions) * 100
            print(f"    Saturation at -1:    {at_minus_one:.2f}%")
            print(f"    Saturation at +1:    {at_plus_one:.2f}%")


def calculate_pid_accuracy(state, df):
    """
    Calculate PID controller accuracy by examining tracking errors.
    State indices 9 and 10 contain roll_error and pitch_error.
    """
    print("\n" + "="*80)
    print("PID CONTROLLER ACCURACY")
    print("="*80)
    
    roll_error = state[:, 9]   # roll_error
    pitch_error = state[:, 10]  # pitch_error
    
    # Convert from radians to degrees for interpretability
    roll_error_deg = np.degrees(roll_error)
    pitch_error_deg = np.degrees(pitch_error)
    
    print(f"\n  ROLL ERROR (target_roll - actual_roll):")
    print(f"    Mean error:          {np.mean(roll_error_deg):10.2f}°")
    print(f"    Std dev:             {np.std(roll_error_deg):10.2f}°")
    print(f"    Root mean square:    {np.sqrt(np.mean(roll_error**2)):10.2f} rad = {np.degrees(np.sqrt(np.mean(roll_error**2))):10.2f}°")
    print(f"    Min error:           {np.min(roll_error_deg):10.2f}°")
    print(f"    Max error:           {np.max(roll_error_deg):10.2f}°")
    
    # Accuracy as percentage within threshold
    within_5deg = np.sum(np.abs(roll_error_deg) < 5.0) / len(roll_error) * 100
    within_10deg = np.sum(np.abs(roll_error_deg) < 10.0) / len(roll_error) * 100
    print(f"    Within ±5°:          {within_5deg:.2f}%")
    print(f"    Within ±10°:         {within_10deg:.2f}%")
    
    print(f"\n  PITCH ERROR (target_pitch - actual_pitch):")
    print(f"    Mean error:          {np.mean(pitch_error_deg):10.2f}°")
    print(f"    Std dev:             {np.std(pitch_error_deg):10.2f}°")
    print(f"    Root mean square:    {np.sqrt(np.mean(pitch_error**2)):10.2f} rad = {np.degrees(np.sqrt(np.mean(pitch_error**2))):10.2f}°")
    print(f"    Min error:           {np.min(pitch_error_deg):10.2f}°")
    print(f"    Max error:           {np.max(pitch_error_deg):10.2f}°")
    
    within_5deg = np.sum(np.abs(pitch_error_deg) < 5.0) / len(pitch_error) * 100
    within_10deg = np.sum(np.abs(pitch_error_deg) < 10.0) / len(pitch_error) * 100
    print(f"    Within ±5°:          {within_5deg:.2f}%")
    print(f"    Within ±10°:         {within_10deg:.2f}%")
    
    # Combined metric
    combined_rms = np.sqrt((np.mean(roll_error**2) + np.mean(pitch_error**2)) / 2)
    print(f"\n  COMBINED TRACKING RMS: {combined_rms:.6f} rad = {np.degrees(combined_rms):.2f}°")


def calculate_state_statistics(state):
    """Calculate state variable statistics."""
    print("\n" + "="*80)
    print("STATE SPACE STATISTICS")
    print("="*80)
    
    for i, name in enumerate(STATE_NAMES):
        values = state[:, i]
        
        # Convert angles to degrees for readability
        if i in [0, 1, 2, 9, 10, 11, 12]:  # angles
            values_display = np.degrees(values)
            unit = "°"
        elif i in [3, 4, 5]:  # angular rates
            values_display = values
            unit = "rad/s"
        else:  # linear velocities and speeds
            values_display = values
            unit = "m/s"
        
        print(f"\n  {name.upper():15} ({unit:>6}):")
        print(f"    Mean:      {np.mean(values_display):12.4f}")
        print(f"    Std:       {np.std(values_display):12.4f}")
        print(f"    Min:       {np.min(values_display):12.4f}")
        print(f"    Max:       {np.max(values_display):12.4f}")
        print(f"    Range:     {np.ptp(values_display):12.4f}")


def calculate_dynamics_statistics(state, action, next_state):
    """Calculate statistics about state transitions and dynamics."""
    print("\n" + "="*80)
    print("DYNAMICS & TRANSITION STATISTICS")
    print("="*80)
    
    # State change magnitude
    state_deltas = next_state - state
    delta_magnitudes = np.linalg.norm(state_deltas, axis=1)
    
    print("\n  STATE CHANGES (||s_t+1 - s_t||):")
    print(f"    Mean magnitude:      {np.mean(delta_magnitudes):12.6f}")
    print(f"    Std dev:             {np.std(delta_magnitudes):12.6f}")
    print(f"    Min:                 {np.min(delta_magnitudes):12.6f}")
    print(f"    Max:                 {np.max(delta_magnitudes):12.6f}")
    print(f"    Median:              {np.median(delta_magnitudes):12.6f}")
    
    # Per-dimension statistics for state changes
    print(f"\n  STATE CHANGE BY DIMENSION:")
    for i, name in enumerate(STATE_NAMES):
        deltas = np.abs(state_deltas[:, i])
        print(f"    {name:15}: mean={np.mean(deltas):10.6f}, std={np.std(deltas):10.6f}, max={np.max(deltas):10.6f}")
    
    # Action effect analysis
    print(f"\n  ACTION STATISTICS:")
    action_magnitudes = np.linalg.norm(action, axis=1)
    print(f"    Mean ||action||:     {np.mean(action_magnitudes):12.6f}")
    print(f"    Std ||action||:      {np.std(action_magnitudes):12.6f}")
    
    # Correlation between action magnitude and state change
    correlation = np.corrcoef(action_magnitudes, delta_magnitudes)[0, 1]
    print(f"    Correlation (action → state change): {correlation:8.4f}")


def calculate_trajectory_statistics(df):
    """Calculate statistics across trajectories."""
    print("\n" + "="*80)
    print("TRAJECTORY STATISTICS")
    print("="*80)
    
    n_trajectories = df['trajectory_id'].nunique()
    steps_per_traj = df['trajectory_id'].value_counts()
    
    print(f"\n  Total trajectories:   {n_trajectories}")
    print(f"  Steps per trajectory: {steps_per_traj.mean():.0f} (all equal)")
    print(f"  Total transitions:    {len(df)}")
    
    # Target combinations
    target_combos = df.groupby(['target_roll', 'target_pitch']).size()
    print(f"\n  Target combinations:  {len(target_combos)}")
    print(f"  Roll targets:         {sorted(df['target_roll'].unique())}")
    print(f"  Pitch targets:        {sorted(df['target_pitch'].unique())}")
    
    # Terminal state frequency
    n_terminal = df['terminal'].sum()
    print(f"\n  Terminal states:      {n_terminal} ({n_terminal/len(df)*100:.2f}%)")
    print(f"  Completed episodes:   {len(df) - n_terminal} ({(len(df) - n_terminal)/len(df)*100:.2f}%)")


def analyze_single_file(csv_path):
    """Analyze a single trajectory file and return key metrics."""
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return None
    
    df = load_trajectory_data(csv_path)
    
    # Parse state/action columns
    state, action, next_state = parse_state_columns(df)
    
    if state is None or action is None or next_state is None:
        print("Error: Could not parse state/action columns!")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Calculate statistics
    calculate_trajectory_statistics(df)
    calculate_reward_stats(df)
    calculate_action_stats(action)
    calculate_pid_accuracy(state, df)
    calculate_state_statistics(state)
    calculate_dynamics_statistics(state, action, next_state)
    
    # Extract key metrics for comparison
    roll_error = state[:, 9]
    pitch_error = state[:, 10]
    roll_error_deg = np.degrees(roll_error)
    pitch_error_deg = np.degrees(pitch_error)
    
    metrics = {
        'name': csv_path.stem.replace('trajectory_data_', '').upper(),
        'env_config': csv_path.stem.replace('trajectory_data_', ''),
        'num_transitions': len(df),
        'mean_reward': np.mean(df['reward'].values),
        'mean_roll_error_deg': np.mean(roll_error_deg),
        'mean_pitch_error_deg': np.mean(pitch_error_deg),
        'std_roll_error_deg': np.std(roll_error_deg),
        'std_pitch_error_deg': np.std(pitch_error_deg),
        'rms_roll_error_deg': np.degrees(np.sqrt(np.mean(roll_error**2))),
        'rms_pitch_error_deg': np.degrees(np.sqrt(np.mean(pitch_error**2))),
        'roll_within_5deg': np.sum(np.abs(roll_error_deg) < 5.0) / len(roll_error) * 100,
        'pitch_within_5deg': np.sum(np.abs(pitch_error_deg) < 5.0) / len(pitch_error) * 100,
        'state': state,
        'action': action,
        'next_state': next_state,
        'df': df
    }
    
    return metrics


def compare_environments(metrics_list):
    """Compare PID performance and dynamics across different environments."""
    print("\n\n" + "="*100)
    print("CROSS-ENVIRONMENT COMPARISON: PID PERFORMANCE & STATE EVOLUTION")
    print("="*100)
    
    # Sort by environment name for consistent order
    metrics_list = sorted(metrics_list, key=lambda x: x['env_config'])
    
    # Comparison table
    print("\n" + " "*15 + "REWARD & PID ACCURACY COMPARISON")
    print("-" * 115)
    print(f"{'Environment':<15} | {'Mean Reward':>12} | {'Roll Error (°)':>20} | {'Pitch Error (°)':>20} | {'Within ±5° (%)':>20}")
    print(f"{'':15} | {'':12} | {'Mean ± Std':>20} | {'Mean ± Std':>20} | {'Roll | Pitch':>20}")
    print("-" * 115)
    
    for m in metrics_list:
        print(f"{m['name']:<15} | {m['mean_reward']:>12.4f} | "
              f"{m['mean_roll_error_deg']:>7.2f} ± {m['std_roll_error_deg']:<6.2f}° | "
              f"{m['mean_pitch_error_deg']:>7.2f} ± {m['std_pitch_error_deg']:<6.2f}° | "
              f"{m['roll_within_5deg']:>6.2f}% | {m['pitch_within_5deg']:>6.2f}%")
    
    # Detailed breakdown
    print("\n" + " "*15 + "RMS TRACKING ERROR (More Robust Metric)")
    print("-" * 80)
    print(f"{'Environment':<15} | {'Roll RMS (°)':>15} | {'Pitch RMS (°)':>15} | {'Combined RMS (°)':>20}")
    print("-" * 80)
    
    for m in metrics_list:
        combined_rms = np.sqrt((m['rms_roll_error_deg']**2 + m['rms_pitch_error_deg']**2) / 2)
        print(f"{m['name']:<15} | {m['rms_roll_error_deg']:>15.2f} | {m['rms_pitch_error_deg']:>15.2f} | {combined_rms:>20.2f}")
    
    # State evolution analysis - show how airspeed and angular velocities change
    print("\n\n" + " "*15 + "STATE EVOLUTION ANALYSIS")
    print("-" * 100)
    print("Comparing how key state variables evolve in each environment:")
    print("\n" + " "*15 + "AIRSPEED (State[13])")
    print("-" * 80)
    print(f"{'Environment':<15} | {'Mean (m/s)':>12} | {'Std (m/s)':>12} | {'Max (m/s)':>12} | {'Min (m/s)':>12}")
    print("-" * 80)
    
    for m in metrics_list:
        airspeed = m['state'][:, 13]
        print(f"{m['name']:<15} | {np.mean(airspeed):>12.2f} | {np.std(airspeed):>12.2f} | {np.max(airspeed):>12.2f} | {np.min(airspeed):>12.2f}")
    
    # Roll rate statistics
    print("\n" + " "*15 + "ROLL RATE - p (State[3]) - rad/s")
    print("-" * 80)
    print(f"{'Environment':<15} | {'Mean (rad/s)':>12} | {'Std (rad/s)':>12} | {'Max (rad/s)':>12} | {'Min (rad/s)':>12}")
    print("-" * 80)
    
    for m in metrics_list:
        p = m['state'][:, 3]
        print(f"{m['name']:<15} | {np.mean(p):>12.4f} | {np.std(p):>12.4f} | {np.max(p):>12.4f} | {np.min(p):>12.4f}")
    
    # Pitch rate statistics
    print("\n" + " "*15 + "PITCH RATE - q (State[4]) - rad/s")
    print("-" * 80)
    print(f"{'Environment':<15} | {'Mean (rad/s)':>12} | {'Std (rad/s)':>12} | {'Max (rad/s)':>12} | {'Min (rad/s)':>12}")
    print("-" * 80)
    
    for m in metrics_list:
        q = m['state'][:, 4]
        print(f"{m['name']:<15} | {np.mean(q):>12.4f} | {np.std(q):>12.4f} | {np.max(q):>12.4f} | {np.min(q):>12.4f}")
    
    # Control effort analysis
    print("\n\n" + " "*15 + "CONTROL EFFORT COMPARISON")
    print("-" * 100)
    print(f"{'Environment':<15} | {'Aileron Mean':>14} | {'Elevator Mean':>14} | {'Throttle Mean':>14} | {'Total Action Mag':>16}")
    print("-" * 100)
    
    for m in metrics_list:
        aileron = m['action'][:, 0]
        elevator = m['action'][:, 1]
        throttle = m['action'][:, 2]
        action_mag = np.linalg.norm(m['action'], axis=1)
        print(f"{m['name']:<15} | {np.mean(aileron):>14.4f} | {np.mean(elevator):>14.4f} | {np.mean(throttle):>14.4f} | {np.mean(action_mag):>16.4f}")
    
    # Analysis summary
    print("\n\n" + "="*100)
    print("INTERPRETATION & INSIGHTS")
    print("="*100)
    
    # Sort by RMS error
    RMS_errors = [(m['name'], np.sqrt((m['rms_roll_error_deg']**2 + m['rms_pitch_error_deg']**2) / 2)) for m in metrics_list]
    RMS_errors.sort(key=lambda x: x[1])
    
    print(f"\n✓ Best PID Performance (Lowest RMS Error): {RMS_errors[0][0]} ({RMS_errors[0][1]:.2f}°)")
    print(f"✓ Most Challenging Environment: {RMS_errors[-1][0]} ({RMS_errors[-1][1]:.2f}°)")
    print(f"  → Difficulty delta: {(RMS_errors[-1][1] - RMS_errors[0][1]):.2f}° RMS difference")
    
    # Airspeed stability
    airspeed_std = [(m['name'], np.std(m['state'][:, 13])) for m in metrics_list]
    airspeed_std.sort(key=lambda x: x[1])
    print(f"\n✓ Most Stable Airspeed: {airspeed_std[0][0]} (σ={airspeed_std[0][1]:.2f} m/s)")
    print(f"✓ Most Variable Airspeed: {airspeed_std[-1][0]} (σ={airspeed_std[-1][1]:.2f} m/s)")
    
    # Reward
    reward_data = [(m['name'], m['mean_reward']) for m in metrics_list]
    reward_data.sort(key=lambda x: x[1], reverse=True)
    print(f"\n✓ Highest Average Reward: {reward_data[0][0]} ({reward_data[0][1]:.4f})")
    print(f"✓ Lowest Average Reward: {reward_data[-1][0]} ({reward_data[-1][1]:.4f})")
    
    print("\nKey Observations:")
    print(f"  • PID controllers show {abs(RMS_errors[-1][1] - RMS_errors[0][1]):.1f}° performance degradation from best to worst")
    print(f"  • Total airspeed variation across environments: {max(a[1] for a in airspeed_std) - min(a[1] for a in airspeed_std):.2f} m/s")
    print(f"  • Each environment file contains 6000 state transitions (3 trajectories × 2000 steps each)")


def main():
    print("\n" + "="*100)
    print("ANALYZING DATA FROM ALL THREE ENVIRONMENTS")
    print("="*100)
    
    # Define the three environment files
    data_dir = Path('../data')
    env_configs = ['noatmo', 'constwind', 'gustsonly']
    metrics_list = []
    
    # Analyze each environment
    for config in env_configs:
        csv_path = data_dir / f'trajectory_data_{config}.csv'
        print(f"\n\n{'#'*100}")
        print(f"# ANALYZING: {config.upper()}")
        print(f"{'#'*100}")
        
        metrics = analyze_single_file(csv_path)
        if metrics is not None:
            metrics_list.append(metrics)
        print("\n")
    
    # Compare all environments
    if len(metrics_list) == len(env_configs):
        compare_environments(metrics_list)
    
    print("\n✓ Cross-environment analysis complete!")

if __name__ == '__main__':
    main()
