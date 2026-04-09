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
# Maximum number of trajectories per environment configuration file
MAX_TRAJECTORIES_PER_FILE = 3

# Define systematic target angles for space coverage (in degrees)
ROLL_TARGETS = [-10, 10, -20, 20, -30, 30, 35, -35]      # 8 roll angles
PITCH_TARGETS = [-10, 10, -20, 20]                       # 4 pitch angles
# Note: Only first MAX_TRAJECTORIES_PER_FILE combinations will be generated

NUM_STEPS = 1000  # 10 seconds at 100 Hz
TRAJECTORIES = []  # Will store trajectory history for future incremental additions

# JSBSim environment configurations to generate data for
#   - 'noatmo': No atmospheric disturbances (baseline)
#   - 'constwind': Constant wind from the north
#   - 'gustsonly': Gusts only (no wind/turbulence)
JSBSIM_CONFIGS = ['noatmo']  


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


def save_trajectory_data_to_csv(trajectory_results, output_file='updated_trajectory_data.csv'):
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


def generate_systematic_trajectories(cfg: DictConfig, jsbsim_config_name='constwind', max_trajectories=3):
    """
    Generate trajectories systematically to cover the observable space.
    
    Args:
        cfg: Hydra configuration
        jsbsim_config_name: Name of the JSBSim config to use (e.g., 'noatmo', 'constwind', 'gustsonly')
        max_trajectories: Maximum number of trajectories to generate for this configuration
    
    Returns:
        List of trajectory result dictionaries
    """
    trajectory_results = []
    
    print("\n" + "="*120)
    print(f"GENERATING TRAJECTORIES FOR ENVIRONMENT: {jsbsim_config_name.upper()}")
    print("="*120)
    print(f"JSBSim config: {jsbsim_config_name}")
    print(f"Max trajectories: {max_trajectories}")
    print(f"Roll targets: {ROLL_TARGETS}")
    print(f"Pitch targets: {PITCH_TARGETS}")
    print("="*120 + "\n")
    
    trajectory_num = 1
    
    # Generate combinations of roll and pitch targets until max_trajectories reached
    for target_roll, target_pitch in itertools.product(ROLL_TARGETS, PITCH_TARGETS):
        if trajectory_num > max_trajectories:
            break
        
        print(f"Loading JSBSim configuration: '{jsbsim_config_name}'")
        cfg.env.jsbsim = OmegaConf.load(f'../config/env/jsbsim/{jsbsim_config_name}.yaml')
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
    """Main function to generate systematic trajectories and save trajectory data for all environment configs."""
    
    try:
        # Loop through each JSBSim environment configuration
        for config_name in JSBSIM_CONFIGS:
            print(f"\n{'='*100}")
            print(f"Processing environment configuration: {config_name.upper()}")
            print(f"{'='*100}")
            
            # Generate trajectories for this configuration
            trajectory_results = generate_systematic_trajectories(
                cfg, 
                jsbsim_config_name=config_name,
                max_trajectories=MAX_TRAJECTORIES_PER_FILE
            )
            
            # Print summary table
            print_trajectory_summary(trajectory_results)
            
            # Save trajectory data to CSV file (CSV format only)
            output_file = f'../data/updated_trajectory_data_{config_name}.csv'
            csv_file = save_trajectory_data_to_csv(trajectory_results, output_file)
        
        print(f"\n{'='*100}")
        print("✓ Data collection complete for all environments!")
        print(f"{'='*100}")
        print(f"\nGenerated 3 CSV files in fw_flightcontrol/data/:")
        for config_name in JSBSIM_CONFIGS:
            print(f"  - updated_trajectory_data_{config_name}.csv")
        print(f"\nYou can load each file with:")
        print(f"  df = pd.read_csv('fw_flightcontrol/data/updated_trajectory_data_noatmo.csv')")

        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_name='default', config_path='../config', version_base=None)
def app(cfg: DictConfig):
    main(cfg)


if __name__ == '__main__':
    app()
