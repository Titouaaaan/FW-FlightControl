#!/usr/bin/env python3
"""
Comprehensive Data Analysis Script
Analyzes trajectory data collected from JSBSim simulator for physics model learning.
Generates publication-quality visualizations and statistical analysis for scientific paper.

Usage:
    python analyze_trajectory_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================
# Get the script directory and navigate to data folder
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR.parent.parent / 'data' / 'trajectory_data_nominal_and_hard_targets.csv'
OUTPUT_DIR = SCRIPT_DIR.parent.parent / 'data' / 'new_analysis_outputs_3.0'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# State dimension names and units
STATE_NAMES = [
    'φ (rad)',      # 0: roll angle
    'θ (rad)',      # 1: pitch angle
    'V_a (m/s)',    # 2: airspeed
    'p (rad/s)',    # 3: roll rate
    'q (rad/s)',    # 4: pitch rate
    'r (rad/s)',    # 5: yaw rate
    'α (rad)',      # 6: angle of attack
    'β (rad)',      # 7: sideslip angle
    'e_φ (rad)',    # 8: roll error
    'e_θ (rad)',    # 9: pitch error
    's10',          # 10
    's11',          # 11
    's12',          # 12
    's13'           # 13
]

ACTION_NAMES = ['δ_a (aileron)', 'δ_e (elevator)', 'δ_t (throttle)']


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_data(csv_file):
    """Load trajectory data from CSV file."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} transitions")
    print(f"✓ Trajectories: {df['trajectory_id'].nunique()}")
    print(f"✓ Columns: {len(df)}")
    return df


def split_trajectories(df):
    """Split dataframe by trajectory ID."""
    trajectories = {}
    for traj_id in sorted(df['trajectory_id'].unique()):
        trajectories[traj_id] = df[df['trajectory_id'] == traj_id].reset_index(drop=True)
    return trajectories


# ============================================================================
# SECTION 1: DATASET OVERVIEW AND BASIC STATISTICS
# ============================================================================
def analyze_dataset_overview(df, trajectories):
    """Generate overview statistics of the dataset."""
    print("\n" + "="*80)
    print("SECTION 1: DATASET OVERVIEW")
    print("="*80)
    
    overview = {
        'Total Transitions': len(df),
        'Number of Trajectories': df['trajectory_id'].nunique(),
        'Avg Transition per Trajectory': len(df) / df['trajectory_id'].nunique(),
        'Unique Roll Targets (deg)': df['target_roll'].nunique(),
        'Unique Pitch Targets (deg)': df['target_pitch'].nunique(),
    }
    
    for key, val in overview.items():
        if isinstance(val, float):
            print(f"{key:.<40} {val:.2f}")
        else:
            print(f"{key:.<40} {val}")
    
    # Statistics per trajectory
    print("\n" + "-"*80)
    print("PER-TRAJECTORY STATISTICS:")
    print("-"*80)
    
    traj_stats = []
    for traj_id in sorted(trajectories.keys()):
        traj = trajectories[traj_id]
        target_roll = traj['target_roll'].iloc[0]
        target_pitch = traj['target_pitch'].iloc[0]
        
        traj_stats.append({
            'Trajectory': traj_id,
            'Target Roll (°)': target_roll,
            'Target Pitch (°)': target_pitch,
            'Steps': len(traj),
            'Terminated': int(traj['terminal'].sum()),
            'Avg Reward': traj['reward'].mean(),
            'Total Reward': traj['reward'].sum(),
        })
    
    stats_df = pd.DataFrame(traj_stats)
    print(stats_df.to_string(index=False))
    
    return stats_df


# ============================================================================
# SECTION 2: STATE TRACKING AND ERROR ANALYSIS
# ============================================================================
def analyze_state_tracking(df, trajectories):
    """Analyze how well the PID controller tracks target states."""
    print("\n" + "="*80)
    print("SECTION 2: STATE TRACKING ANALYSIS (PID Controller Performance)")
    print("="*80)
    
    # Extract state columns
    state_cols = [f's_t_{i}' for i in range(14)]
    next_state_cols = [f's_t+1_{i}' for i in range(14)]
    
    tracking_data = []
    
    for traj_id in sorted(trajectories.keys()):
        traj = trajectories[traj_id]
        target_roll = np.deg2rad(traj['target_roll'].iloc[0])
        target_pitch = np.deg2rad(traj['target_pitch'].iloc[0])
        
        # Extract roll (state 0) and pitch (state 1)
        roll_angles = traj['s_t_0'].values
        pitch_angles = traj['s_t_1'].values
        
        # Compute tracking errors
        roll_error = np.abs(roll_angles - target_roll)
        pitch_error = np.abs(pitch_angles - target_pitch)
        
        tracking_data.append({
            'Trajectory': traj_id,
            'Target Roll (deg)': np.rad2deg(target_roll),
            'Target Pitch (deg)': np.rad2deg(target_pitch),
            'Avg Roll Error (deg)': np.rad2deg(np.mean(roll_error)),
            'Max Roll Error (deg)': np.rad2deg(np.max(roll_error)),
            'Avg Pitch Error (deg)': np.rad2deg(np.mean(pitch_error)),
            'Max Pitch Error (deg)': np.rad2deg(np.max(pitch_error)),
            'Avg Roll (deg)': np.rad2deg(np.mean(roll_angles)),
            'Avg Pitch (deg)': np.rad2deg(np.mean(pitch_angles)),
            'Std Roll (deg)': np.rad2deg(np.std(roll_angles)),
            'Std Pitch (deg)': np.rad2deg(np.std(pitch_angles)),
        })
    
    tracking_df = pd.DataFrame(tracking_data)
    print("\n" + tracking_df.to_string(index=False))
    
    return tracking_df


# ============================================================================
# SECTION 3: STATE VARIANCE ANALYSIS AT DIFFERENT TIME SCALES
# ============================================================================
def analyze_state_variance(trajectories):
    """Analyze state variance at different time scales (windowing)."""
    print("\n" + "="*80)
    print("SECTION 3: STATE VARIANCE ANALYSIS AT DIFFERENT TIME SCALES")
    print("="*80)
    
    state_cols = [f's_t_{i}' for i in range(14)]
    
    # Time scales to analyze (in steps)
    time_scales = [1, 5, 10, 20, 50, 100]
    
    print("\nAnalyzing variance at different time scales...")
    print("(Higher variance = more dynamic behavior at that scale)\n")
    
    variance_results = {ts: {i: [] for i in range(14)} for ts in time_scales}
    
    for traj_id, traj in trajectories.items():
        for state_idx in range(14):
            state_col = state_cols[state_idx]
            state_values = traj[state_col].values
            
            for ts in time_scales:
                if len(state_values) > ts:
                    # Compute variance of state differences over time scale ts
                    diffs = np.diff(state_values, n=1)[::ts]
                    var = np.var(diffs) if len(diffs) > 0 else 0
                    variance_results[ts][state_idx].append(var)
    
    # Create variance summary table
    variance_summary = []
    for ts in time_scales:
        row = {'Time Scale (steps)': ts}
        for state_idx in range(8):  # Only first 8 states (main ones)
            mean_var = np.mean(variance_results[ts][state_idx])
            row[f'State {state_idx}'] = mean_var
        variance_summary.append(row)
    
    variance_df = pd.DataFrame(variance_summary)
    print(variance_df.to_string(index=False))
    
    return variance_results, variance_df


# ============================================================================
# SECTION 4: ACTION STATISTICS
# ============================================================================
def analyze_actions(df, trajectories):
    """Analyze action statistics and distributions."""
    print("\n" + "="*80)
    print("SECTION 4: ACTION STATISTICS")
    print("="*80)
    
    action_cols = [f'a_t_{i}' for i in range(3)]
    
    action_stats = []
    for traj_id in sorted(trajectories.keys()):
        traj = trajectories[traj_id]
        for i, action_name in enumerate(ACTION_NAMES):
            action_col = action_cols[i]
            action_stats.append({
                'Trajectory': traj_id,
                'Action': action_name,
                'Mean': traj[action_col].mean(),
                'Std': traj[action_col].std(),
                'Min': traj[action_col].min(),
                'Max': traj[action_col].max(),
                'Range': traj[action_col].max() - traj[action_col].min(),
            })
    
    action_stats_df = pd.DataFrame(action_stats)
    
    print("\n" + action_stats_df.to_string(index=False))
    
    # Overall statistics
    print("\n" + "-"*80)
    print("OVERALL ACTION STATISTICS:")
    print("-"*80)
    overall_stats = []
    for i, action_name in enumerate(ACTION_NAMES):
        action_col = action_cols[i]
        overall_stats.append({
            'Action': action_name,
            'Mean': df[action_col].mean(),
            'Std': df[action_col].std(),
            'Min': df[action_col].min(),
            'Max': df[action_col].max(),
            'Q25': df[action_col].quantile(0.25),
            'Median': df[action_col].median(),
            'Q75': df[action_col].quantile(0.75),
        })
    
    overall_df = pd.DataFrame(overall_stats)
    print("\n" + overall_df.to_string(index=False))
    
    return action_stats_df, overall_df


# ============================================================================
# SECTION 5: STATE TRANSITION ANALYSIS
# ============================================================================
def analyze_state_transitions(df, trajectories):
    """Analyze state transition magnitudes."""
    print("\n" + "="*80)
    print("SECTION 5: STATE TRANSITION ANALYSIS")
    print("="*80)
    
    state_cols = [f's_t_{i}' for i in range(14)]
    next_state_cols = [f's_t+1_{i}' for i in range(14)]
    
    transition_stats = []
    
    for state_idx in range(8):  # Focus on main 8 states
        state_col = state_cols[state_idx]
        next_state_col = next_state_cols[state_idx]
        
        # Compute transition magnitude (state change per step)
        state_deltas = np.abs(df[next_state_col].values - df[state_col].values)
        
        transition_stats.append({
            'State': STATE_NAMES[state_idx],
            'Mean Δ': np.mean(state_deltas),
            'Std Δ': np.std(state_deltas),
            'Min Δ': np.min(state_deltas),
            'Max Δ': np.max(state_deltas),
            'Median Δ': np.median(state_deltas),
            'Q90 Δ': np.quantile(state_deltas, 0.90),
        })
    
    transition_df = pd.DataFrame(transition_stats)
    print("\n" + transition_df.to_string(index=False))
    
    return transition_df


# ============================================================================
# VISUALIZATION: STATE EVOLUTION OVER TIME
# ============================================================================
def plot_state_evolution(trajectories):
    """Plot state evolution for all trajectories."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION: State Evolution Over Time")
    print("="*80)
    
    state_cols = [f's_t_{i}' for i in range(8)]
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Color map for trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for state_idx, (ax, state_col) in enumerate(zip(axes, state_cols)):
        for (traj_id, traj), color in zip(trajectories.items(), colors):
            states = traj[state_col].values
            ax.plot(states, alpha=0.7, linewidth=1.2, color=color, label=f'Trajectory {traj_id}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(STATE_NAMES[state_idx])
        ax.set_title(f'State {state_idx}: {STATE_NAMES[state_idx]}')
        ax.grid(True, alpha=0.3)
    
    # Create legend at bottom center outside plot
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                                 label=f'Trajectory {i+1}')
                      for i in range(len(trajectories))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01),
              ncol=len(trajectories), fontsize=10)
    
    fig.suptitle('State Evolution Over Time', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'state_evolution_over_time.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: state_evolution_over_time.png")
    plt.close()


# ============================================================================
# VISUALIZATION: ACTION DISTRIBUTIONS
# ============================================================================
def plot_action_distributions(df, trajectories):
    """Plot action distributions across all trajectories and per trajectory."""
    print("\nGENERATING VISUALIZATION: Action Distributions")
    
    action_cols = [f'a_t_{i}' for i in range(3)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # Overall distribution + per-trajectory distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for ax_idx, (action_col, action_name) in enumerate(zip(action_cols, ACTION_NAMES)):
        # Row 1: Overall distribution
        ax = axes[0, ax_idx]
        ax.hist(df[action_col].values, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
        ax.set_xlabel(f'{action_name} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{action_name} - Overall Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Row 2: By trajectory
        ax = axes[1, ax_idx]
        for (traj_id, traj), color in zip(trajectories.items(), colors):
            ax.hist(traj[action_col].values, bins=30, alpha=0.5, label=f'Traj {traj_id}',
                   edgecolor='black', linewidth=0.5, color=color)
        
        ax.set_xlabel(f'{action_name} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{action_name} - Per Trajectory')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Create legend at bottom center outside plot
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                                 label=f'Trajectory {i+1}')
                      for i in range(len(trajectories))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=len(trajectories), fontsize=10)
    
    fig.suptitle('Distribution of Control Actions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'action_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: action_distributions.png")
    plt.close()


# ============================================================================
# VISUALIZATION: REWARD OVER TIME
# ============================================================================
def plot_reward_evolution(trajectories):
    """Plot reward evolution for all trajectories."""
    print("\nGENERATING VISUALIZATION: Reward Evolution")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for (traj_id, traj), color in zip(trajectories.items(), colors):
        # Compute cumulative reward
        cumulative_reward = np.cumsum(traj['reward'].values)
        ax.plot(cumulative_reward, linewidth=1.5, alpha=0.7, color=color, label=f'Trajectory {traj_id}')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Evolution')
    ax.grid(True, alpha=0.3)
    
    # Create legend at bottom center outside plot
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                                 label=f'Trajectory {i+1}')
                      for i in range(len(trajectories))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=len(trajectories), fontsize=10)
    
    fig.suptitle('Cumulative Reward Evolution Over Trajectory', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'reward_evolution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: reward_evolution.png")
    plt.close()


# ============================================================================
# VISUALIZATION: TRACKING ERROR OVER TIME
# ============================================================================
def plot_tracking_errors(trajectories):
    """Plot roll and pitch tracking errors over time."""
    print("\nGENERATING VISUALIZATION: Tracking Errors Over Time")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for idx, (traj_id, traj) in enumerate(trajectories.items()):
        # For progressive targets, compute error at each step using current target
        roll_angles = traj['s_t_0'].values
        pitch_angles = traj['s_t_1'].values
        target_roll_deg = traj['target_roll'].values
        target_pitch_deg = traj['target_pitch'].values
        
        # Convert to radians for computation
        roll_angles_deg = np.rad2deg(roll_angles)
        pitch_angles_deg = np.rad2deg(pitch_angles)
        
        roll_error_deg = np.abs(roll_angles_deg - target_roll_deg)
        pitch_error_deg = np.abs(pitch_angles_deg - target_pitch_deg)
        
        color = colors[idx]
        label = f'Trajectory {traj_id}'
        
        # Roll error
        axes[0].plot(roll_error_deg, linewidth=1.2, alpha=0.7, color=color, label=label)
        
        # Pitch error
        axes[1].plot(pitch_error_deg, linewidth=1.2, alpha=0.7, color=color, label=label)
    
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Absolute Error (degrees)')
    axes[0].set_title('Roll Angle Tracking Error')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Absolute Error (degrees)')
    axes[1].set_title('Pitch Angle Tracking Error')
    axes[1].grid(True, alpha=0.3)
    
    # Create legend at bottom center outside plot
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                                 label=f'Trajectory {i+1}')
                      for i in range(len(trajectories))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=len(trajectories), fontsize=10)
    
    fig.suptitle('Attitude Control Tracking Errors', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tracking_errors_over_time.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tracking_errors_over_time.png")
    plt.close()


# ============================================================================
# VISUALIZATION: STATE TRANSITION MAGNITUDE BY TIME WINDOW
# ============================================================================
def plot_transition_magnitudes(df):
    """Plot mean state transition magnitudes binned by trajectory time windows."""
    print("\nGENERATING VISUALIZATION: State Transition Magnitude by Time Window")
    
    state_cols = [f's_t_{i}' for i in range(8)]
    next_state_cols = [f's_t+1_{i}' for i in range(8)]
    
    # Define time windows (bins of 100 steps)
    time_windows = [(0, 100), (100, 200), (200, 300), (300, 400), 
                    (400, 500), (500, 600), (600, 700), (700, 800),
                    (800, 900), (900, 1000)]
    window_labels = [f"{s}-{e}" for s, e in time_windows]
    
    # Compute mean transitions for each time window and state
    transition_matrix = np.zeros((8, len(time_windows)))
    
    for window_idx, (start, end) in enumerate(time_windows):
        # Filter data by step_id
        window_data = df[(df['step_id'] >= start) & (df['step_id'] < end)]
        
        for state_idx in range(8):
            state_deltas = np.abs(window_data[next_state_cols[state_idx]].values - 
                                 window_data[state_cols[state_idx]].values)
            transition_matrix[state_idx, window_idx] = np.mean(state_deltas)
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    
    # Use better colormap with clear visual hierarchy
    im = ax.imshow(transition_matrix, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest', vmin=0)
    
    # Set ticks and labels
    ax.set_xticks(range(len(time_windows)))
    ax.set_xticklabels(window_labels, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticks(range(8))
    ax.set_yticklabels([STATE_NAMES[i] for i in range(8)], fontsize=12, weight='bold')
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(time_windows)) - 0.5, minor=True)
    ax.set_yticks(np.arange(8) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Labels
    ax.set_xlabel('Trajectory Phase (steps)', fontsize=13, weight='bold', labelpad=12)
    ax.set_ylabel('State Dimension', fontsize=13, weight='bold', labelpad=12)
    ax.set_title('Mean Absolute State Changes by Trajectory Phase\n(Showing when dynamics are most active)', 
                fontsize=14, weight='bold', pad=20)
    
    # Colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Mean |ΔState| per timestep', rotation=270, labelpad=25, fontsize=11, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations with better styling
    for i in range(8):
        for j in range(len(time_windows)):
            value = transition_matrix[i, j]
            # Choose text color based on background brightness
            text_color = 'white' if value > np.max(transition_matrix) * 0.6 else 'black'
            ax.text(j, i, f'{value:.5f}',
                   ha="center", va="center", color=text_color, 
                   fontsize=9, weight='bold', family='monospace')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'state_transition_distributions.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✓ Saved: state_transition_distributions.png")
    plt.close()


# ============================================================================
# VISUALIZATION: ROLLOUT STABILITY (CVaR analysis)
# ============================================================================
def plot_rollout_stability(trajectories):
    """Analyze and plot trajectory stability for all trajectories."""
    print("\nGENERATING VISUALIZATION: Rollout Stability Analysis")
    
    state_cols = [f's_t_{i}' for i in range(8)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for state_idx, (ax, state_col) in enumerate(zip(axes, state_cols)):
        for (traj_id, traj), color in zip(trajectories.items(), colors):
            states = traj[state_col].values
            # Compute rolling standard deviation over windows of 20 steps
            window = 20
            rolling_std = pd.Series(states).rolling(window=window).std().values
            ax.plot(rolling_std, linewidth=1.2, alpha=0.7, color=color, label=f'Trajectory {traj_id}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Rolling Std Dev (window=20)')
        ax.set_title(f'{STATE_NAMES[state_idx]} - Stability')
        ax.grid(True, alpha=0.3)
    
    # Create legend at bottom center outside plot
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=2,
                                 label=f'Trajectory {i+1}')
                      for i in range(len(trajectories))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
              ncol=len(trajectories), fontsize=10)
    
    fig.suptitle('Flight State Stability Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rollout_stability.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: rollout_stability.png")
    plt.close()




# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("TRAJECTORY DATA ANALYSIS - COMPREHENSIVE RESEARCH ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_data(CSV_FILE)
    trajectories = split_trajectories(df)
    
    # Section 1: Dataset Overview
    overview_stats = analyze_dataset_overview(df, trajectories)
    
    # Section 2: State Tracking
    tracking_df = analyze_state_tracking(df, trajectories)
    
    # Section 3: Variance Analysis
    variance_results, variance_df = analyze_state_variance(trajectories)
    
    # Section 4: Action Statistics
    action_stats, action_overall = analyze_actions(df, trajectories)
    
    # Section 5: State Transitions
    transition_df = analyze_state_transitions(df, trajectories)
    
    # Generate all visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_state_evolution(trajectories)
    plot_action_distributions(df, trajectories)
    plot_reward_evolution(trajectories)
    plot_tracking_errors(trajectories)
    plot_transition_magnitudes(df)
    plot_rollout_stability(trajectories)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("\nGenerated plots:")
    print("  - state_evolution_over_time.png")
    print("  - action_distributions.png")
    print("  - reward_evolution.png")
    print("  - tracking_errors_over_time.png")
    print("  - state_transition_distributions.png")
    print("  - rollout_stability.png")


if __name__ == '__main__':
    main()
