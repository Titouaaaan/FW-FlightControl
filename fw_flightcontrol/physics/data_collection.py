"""
Trajectory Data Collection Script
Generates trajectories systematically to cover the observable space.
Records state transitions (s_t, a_t, s_t+1, r_t) for dynamics learning.

Usage:
    python data_collection.py
"""

import numpy as np
import gymnasium as gym
import fw_jsbgym
import hydra
from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.utils import jsbsim_properties as prp
from omegaconf import DictConfig, OmegaConf
import itertools
import csv
import os
from pathlib import Path



# ============================================================================
# CONFIGURATION
# ============================================================================
# Define systematic target angles for space coverage (in degrees)
ROLL_TARGETS = [-10, 10, -20, 20, -30, 30, 35, -35]      # 8 roll angles
PITCH_TARGETS = [-10,10 , -20, 20]                      # 4 pitch angles
# Total trajectories: 8 * 4 = 32 (easily extensible by adding more values)

NUM_STEPS = 2000  # 20 seconds at 100 Hz
TRAJECTORIES = []  # Will store trajectory history for future incremental additions

#   - 'noatmo': No atmospheric disturbances (baseline)
#   - 'constwind': Constant wind from the north
#   - 'gustsonly': Gusts only (no wind/turbulence)
#   - 'turbonly': Turbulence only
#   - 'alldist': All disturbances combined
JSBSIM_CONFIG = 'constwind'  


def run_single_trajectory(env, trajectory_num, target_roll_deg, target_pitch_deg):
    """
    Run a single attitude control trajectory with PID controllers.
    
    Args:
        env: The attitude control environment
        trajectory_num: Trajectory number for display
        target_roll_deg: Target roll angle in degrees
        target_pitch_deg: Target pitch angle in degrees
    
    Returns:
        Dictionary with:
        - Trajectory metadata (target_roll, target_pitch, etc.)
        - List of transitions: [(s_t, a_t, s_t+1, r_t), ...]
    """
    
    # Convert targets to radians
    target_roll_rad = np.deg2rad(target_roll_deg)
    target_pitch_rad = np.deg2rad(target_pitch_deg)
    
    # Initialize PID controllers
    pid_aileron = PID(
        kp=1.5, ki=0.1, kd=0.1,
        dt=env.unwrapped.fdm_dt,
        trim=TrimPoint(),
        limit=1.0,
        is_throttle=False
    )
    
    pid_elevator = PID(
        kp=-2.0, ki=-0.3, kd=-0.1,
        dt=env.unwrapped.fdm_dt,
        trim=TrimPoint(),
        limit=1.0,
        is_throttle=False
    )
    
    # Initialize environment
    env.unwrapped.init()
    obs, info = env.reset()
    
    # Set target state
    target_state = np.array([target_roll_rad, target_pitch_rad])
    env.set_target_state(target_state)
    
    # Run trajectory and collect transitions
    episode_reward = 0.0
    transitions = []  # List of (s_t, a_t, s_t+1, r_t, terminal_flag)
    roll_angles = []
    pitch_angles = []
    roll_errors = []
    pitch_errors = []
    
    for step in range(NUM_STEPS):
        # Store current state (s_t)
        state_t = obs.copy()
        
        # Extract state for PID
        roll = obs[0]
        pitch = obs[1]
        p_radps = obs[3]  # roll rate
        q_radps = obs[4]  # pitch rate
        
        # Update PID references and compute commands
        pid_aileron.set_reference(target_roll_rad)
        pid_elevator.set_reference(target_pitch_rad)
        
        aileron_cmd, _, _ = pid_aileron.update(state=roll, state_dot=p_radps, saturate=True, normalize=False)
        elevator_cmd, _, _ = pid_elevator.update(state=pitch, state_dot=q_radps, saturate=True, normalize=False)
        
        # Store action (a_t) - will be updated with throttle after step
        action = np.array([aileron_cmd, elevator_cmd, 0.0])  # throttle added below
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action[:2])  # Only send aileron/elevator to env
        episode_reward += reward
        
        # Query throttle from environment after step
        throttle = env.unwrapped.sim[prp.throttle_cmd]
        action[2] = throttle  # Update 3rd dimension with actual throttle
        
        # Store next state (s_t+1) and transition info
        state_next = obs.copy()
        terminal = terminated or truncated
        
        # Record transition: (s_t, a_t, s_t+1, r_t, terminal)
        transitions.append({
            'state_t': state_t,
            'action': action,
            'state_next': state_next,
            'reward': reward,
            'terminal': terminal
        })
        
        # Track angles and errors for statistics
        roll_angles.append(np.rad2deg(obs[0]))
        pitch_angles.append(np.rad2deg(obs[1]))
        
        if len(obs) > 7:
            roll_errors.append(np.rad2deg(obs[6]))
            pitch_errors.append(np.rad2deg(obs[7]))
        else:
            roll_errors.append(np.rad2deg(target_roll_rad - obs[0]))
            pitch_errors.append(np.rad2deg(target_pitch_rad - obs[1]))
        
        if terminal:
            break
    
    env.close()
    
    # Compute final statistics
    avg_roll = np.mean(roll_angles) if roll_angles else 0.0
    avg_pitch = np.mean(pitch_angles) if pitch_angles else 0.0
    avg_roll_error = np.mean(np.abs(roll_errors)) if roll_errors else 0.0
    avg_pitch_error = np.mean(np.abs(pitch_errors)) if pitch_errors else 0.0
    
    trajectory_data = {
        'metadata': {
            'trajectory_num': trajectory_num,
            'target_roll': target_roll_deg,
            'target_pitch': target_pitch_deg,
            'avg_roll': avg_roll,
            'avg_pitch': avg_pitch,
            'avg_roll_error': avg_roll_error,
            'avg_pitch_error': avg_pitch_error,
            'total_reward': episode_reward,
            'steps_executed': len(roll_angles)
        },
        'transitions': transitions
    }
    
    print(f"[Traj {trajectory_num:2d}] Targets: Roll={target_roll_deg:5.1f}°, Pitch={target_pitch_deg:5.1f}° → "
          f"Avg Roll={avg_roll:6.2f}°, Avg Pitch={avg_pitch:6.2f}°, "
          f"Errors: Roll={avg_roll_error:5.2f}°, Pitch={avg_pitch_error:5.2f}° | "
          f"Transitions: {len(transitions)}")
    
    return trajectory_data


def save_trajectory_data_to_csv(trajectory_results, output_file='trajectory_data.csv'):
    """
    Save trajectory data (state transitions) to CSV file for dynamics learning.
    
    CSV format:
        trajectory_id, step_id, target_roll, target_pitch,
        s_t_0, s_t_1, ..., s_t_13 (state at time t),
        a_t_0, a_t_1 (action at time t: aileron, elevator),
        s_t+1_0, s_t+1_1, ..., s_t+1_13 (next state),
        reward, terminal
    
    Args:
        trajectory_results: List of trajectory data dictionaries
        output_file: Path to save CSV file
    """
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_transitions = 0
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'trajectory_id', 'step_id', 'target_roll', 'target_pitch'
        ]
        
        # Add state dimensions
        for i in range(14):
            fieldnames.append(f's_t_{i}')
        
        # Add action dimensions
        for i in range(3):
            fieldnames.append(f'a_t_{i}')
        
        # Add next state dimensions
        for i in range(14):
            fieldnames.append(f's_t+1_{i}')
        
        # Add reward and terminal flag
        fieldnames.extend(['reward', 'terminal'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write all transitions from all trajectories
        for traj_data in trajectory_results:
            meta = traj_data['metadata']
            trajectory_id = meta['trajectory_num']
            target_roll = meta['target_roll']
            target_pitch = meta['target_pitch']
            
            for step_id, transition in enumerate(traj_data['transitions']):
                row = {
                    'trajectory_id': trajectory_id,
                    'step_id': step_id,
                    'target_roll': target_roll,
                    'target_pitch': target_pitch
                }
                
                # Add state values
                state_t = transition['state_t']
                for i in range(14):
                    row[f's_t_{i}'] = state_t[i]
                
                # Add action values
                action = transition['action']
                for i in range(3):
                    row[f'a_t_{i}'] = action[i]
                
                # Add next state values
                state_next = transition['state_next']
                for i in range(14):
                    row[f's_t+1_{i}'] = state_next[i]
                
                # Add reward and terminal flag
                row['reward'] = transition['reward']
                row['terminal'] = 1 if transition['terminal'] else 0
                
                writer.writerow(row)
                total_transitions += 1
    
    print(f"\n✓ Saved {total_transitions} state transitions to '{output_file}'")
    print(f"  CSV format: trajectory_id, step_id, target_roll, target_pitch, ")
    print(f"              s_t_0-13 (state), a_t_0-2 (action), s_t+1_0-13 (next state), reward, terminal")
    return output_file


def save_trajectory_data_to_parquet(trajectory_results, output_file='trajectory_data.parquet'):
    """
    Save trajectory data to Parquet format (optimized for ML/dynamics learning).
    
    Parquet format (efficient columnar storage with nested arrays):
        trajectory_id, step_id, target_roll, target_pitch,
        state (array[14]), action (array[3]), next_state (array[14]),
        reward, terminal
    
    Benefits:
    - 5x compression vs CSV (~3-4MB vs 17MB for 42k transitions)
    - Native array support (state/action/next_state as arrays, not 30 separate columns)
    - Columnar storage (optimal for ML training)
    - Widely supported (pandas, PyTorch, TensorFlow, DuckDB, Spark)
    
    Args:
        trajectory_results: List of trajectory data dictionaries
        output_file: Path to save Parquet file
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available for Parquet output")
        return None
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all transitions into lists
    data = {
        'trajectory_id': [],
        'step_id': [],
        'target_roll': [],
        'target_pitch': [],
        'state': [],
        'action': [],
        'next_state': [],
        'reward': [],
        'terminal': []
    }
    
    total_transitions = 0
    
    for traj_data in trajectory_results:
        meta = traj_data['metadata']
        trajectory_id = meta['trajectory_num']
        target_roll = meta['target_roll']
        target_pitch = meta['target_pitch']
        
        for step_id, transition in enumerate(traj_data['transitions']):
            data['trajectory_id'].append(trajectory_id)
            data['step_id'].append(step_id)
            data['target_roll'].append(target_roll)
            data['target_pitch'].append(target_pitch)
            
            # Store as numpy arrays (will be nested in Parquet)
            data['state'].append(np.array(transition['state_t'], dtype=np.float32))
            data['action'].append(np.array(transition['action'], dtype=np.float32))
            data['next_state'].append(np.array(transition['state_next'], dtype=np.float32))
            
            data['reward'].append(transition['reward'])
            data['terminal'].append(1 if transition['terminal'] else 0)
            
            total_transitions += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Parquet with compression
    df.to_parquet(output_file, compression='snappy', index=False)
    
    # Get file size for display
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
    
    print(f"\n✓ Saved {total_transitions} state transitions to '{output_file}' ({file_size:.2f}MB)")
    print(f"  Parquet format (optimized for ML):")
    print(f"    - Nested arrays: state[14], action[2], next_state[14]")
    print(f"    - Compression: snappy (5x smaller than CSV)")
    print(f"    - Columnar storage (optimal for training)")
    
    return output_file


def print_trajectory_summary(trajectory_results):
    """Print a formatted table of trajectory results."""
    print("\n" + "="*140)
    print("TRAJECTORY SUMMARY")
    print("="*140)
    print(f"{'Traj':>4} | {'Target Roll':>12} | {'Target Pitch':>13} | {'Avg Roll':>10} | {'Avg Pitch':>11} | "
          f"{'Roll Error':>11} | {'Pitch Error':>12} | {'Reward':>10} | {'Steps':>6} | {'Transitions':>12}")
    print("-" * 140)
    
    for traj in trajectory_results:
        meta = traj['metadata']
        num_transitions = len(traj['transitions'])
        
        print(f"{meta['trajectory_num']:4d} | {meta['target_roll']:12.2f}° | {meta['target_pitch']:13.2f}° | "
              f"{meta['avg_roll']:10.2f}° | {meta['avg_pitch']:11.2f}° | "
              f"{meta['avg_roll_error']:11.2f}° | {meta['avg_pitch_error']:12.2f}° | "
              f"{meta['total_reward']:10.4f} | {meta['steps_executed']:6d} | {num_transitions:12d}")
    
    print("-" * 140)
    
    # Compute and print averages
    num_trajectories = len(trajectory_results)
    total_transitions = sum(len(t['transitions']) for t in trajectory_results)
    avg_target_roll = np.mean([t['metadata']['target_roll'] for t in trajectory_results])
    avg_target_pitch = np.mean([t['metadata']['target_pitch'] for t in trajectory_results])
    avg_obs_roll = np.mean([t['metadata']['avg_roll'] for t in trajectory_results])
    avg_obs_pitch = np.mean([t['metadata']['avg_pitch'] for t in trajectory_results])
    avg_roll_err = np.mean([t['metadata']['avg_roll_error'] for t in trajectory_results])
    avg_pitch_err = np.mean([t['metadata']['avg_pitch_error'] for t in trajectory_results])
    avg_reward = np.mean([t['metadata']['total_reward'] for t in trajectory_results])
    
    print(f"{'AVG':>4} | {avg_target_roll:12.2f}° | {avg_target_pitch:13.2f}° | "
          f"{avg_obs_roll:10.2f}° | {avg_obs_pitch:11.2f}° | "
          f"{avg_roll_err:11.2f}° | {avg_pitch_err:12.2f}° | "
          f"{avg_reward:10.4f} |")
    print("="*140 + "\n")
    
    print(f"Total trajectories generated: {num_trajectories}")
    print(f"Total state transitions collected: {total_transitions}")
    print(f"Configuration: ROLL_TARGETS = {ROLL_TARGETS}")
    print(f"Configuration: PITCH_TARGETS = {PITCH_TARGETS}")
    print(f"(To extend coverage, add more values to ROLL_TARGETS or PITCH_TARGETS at the top of the script)\n")


def generate_systematic_trajectories(cfg: DictConfig):
    """
    Generate trajectories systematically to cover the observable space.
    
    Args:
        cfg: Hydra configuration
    
    Returns:
        List of trajectory result dictionaries
    """
    trajectory_results = []
    
    print("\n" + "="*120)
    print(f"GENERATING SYSTEMATIC TRAJECTORIES FOR SPACE COVERAGE")
    print("="*120)
    print(f"Nominal conditions (no wind or disturbances)")
    print(f"Roll targets: {ROLL_TARGETS}")
    print(f"Pitch targets: {PITCH_TARGETS}")
    print(f"Number of trajectories: {len(ROLL_TARGETS) * len(PITCH_TARGETS)}")
    print("="*120 + "\n")
    
    trajectory_num = 1
    
    # Generate all combinations of roll and pitch targets
    for target_roll, target_pitch in itertools.product(ROLL_TARGETS, PITCH_TARGETS):
        print(f"Loading JSBSim configuration: '{JSBSIM_CONFIG}'")
        cfg.env.jsbsim = OmegaConf.load(f'config/env/jsbsim/{JSBSIM_CONFIG}.yaml')
        try:
            # Create fresh environment for each trajectory
            env = gym.make(
                'ACBohnNoVaIErr-v0',
                cfg_env=cfg.env,
                render_mode='none'
            )
            
            # Run trajectory
            trajectory_data = run_single_trajectory(
                env, trajectory_num, target_roll, target_pitch
            )
            
            trajectory_results.append(trajectory_data)
            TRAJECTORIES.append(trajectory_data)  # Store for future incremental additions
            
            trajectory_num += 1
            
        except Exception as e:
            print(f"Error in trajectory {trajectory_num}: {e}")
            import traceback
            traceback.print_exc()
    
    return trajectory_results


def main(cfg: DictConfig):
    """Main function to generate systematic trajectories and save trajectory data."""
    
    try:
        # Generate trajectories with systematic coverage
        trajectory_results = generate_systematic_trajectories(cfg)
        
        # Print summary table
        print_trajectory_summary(trajectory_results)
        
        # Save trajectory data to both CSV and Parquet
        csv_file = save_trajectory_data_to_csv(trajectory_results, '../data/trajectory_data.csv')
        parquet_file = save_trajectory_data_to_parquet(trajectory_results, '../data/trajectory_data.parquet')
        
        print(f"\n{'='*80}")
        print("✓ Data collection complete!")
        print(f"{'='*80}")
        print(f"CSV format (for inspection):    {csv_file}")
        print(f"Parquet format (for training): {parquet_file}")
        print(f"\nYou can now use either format for dynamics learning:")
        print(f"  Python: df = pd.read_parquet('trajectory_data.parquet')")
        print(f"  Or:     df = pd.read_csv('trajectory_data.csv')")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_name='default', config_path='config', version_base=None)
def app(cfg: DictConfig):
    main(cfg)


if __name__ == '__main__':
    app()
