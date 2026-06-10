"""
Updated Trajectory Data Collection Script with Progressive Target Angles
Generates trajectories with time-varying target angles that change smoothly every 250 steps.
Records state transitions (s_t, a_t, s_t+1, r_t) for dynamics learning.

Usage:
    python updated_data_collection.py
"""

import numpy as np
import gymnasium as gym
import fw_jsbgym
import hydra
from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.utils import jsbsim_properties as prp
from omegaconf import DictConfig, OmegaConf
import csv
import os
from pathlib import Path



# ============================================================================
# CONFIGURATION
# ============================================================================
# Trajectory parameters
NUM_STEPS = 4000                           # 40 seconds at 100 Hz

# Target angle bounds — limited to range where PID can converge cleanly.
ROLL_MIN, ROLL_MAX = -55, 55
PITCH_MIN, PITCH_MAX = -28, 28

# Maximum change per target jump (degrees)
MAX_DELTA_ROLL = 25
MAX_DELTA_PITCH = 20

# Convergence criterion: both roll AND pitch must be within this tolerance
# for CONVERGENCE_STEPS consecutive steps before the target is allowed to change.
CONVERGENCE_TOL_DEG   = 1.5
CONVERGENCE_STEPS     = 100

# Number of trajectories to generate
NUM_TRAJECTORIES = 150

# Persistent excitation: Gaussian noise added to PID aileron/elevator outputs.
EXCITATION_NOISE_STD = 0.0

# JSBSim environment configurations to generate data for
JSBSIM_CONFIGS = ['noatmo']


def generate_progressive_targets(num_intervals=20,
                                 max_delta_roll=MAX_DELTA_ROLL,
                                 max_delta_pitch=MAX_DELTA_PITCH,
                                 roll_min=ROLL_MIN, roll_max=ROLL_MAX,
                                 pitch_min=PITCH_MIN, pitch_max=PITCH_MAX):
    """
    Generate progressive target angles that change smoothly every interval.

    Uses a random walk approach to ensure smooth transitions between targets.

    Args:
        num_intervals: Number of intervals
        max_delta_roll: Maximum change in roll angle per interval (degrees)
        max_delta_pitch: Maximum change in pitch angle per interval (degrees)
        roll_min, roll_max: Bounds for roll angle
        pitch_min, pitch_max: Bounds for pitch angle

    Returns:
        Tuple of (roll_targets, pitch_targets) arrays of length num_intervals
    """
    # Initialize with random starting points within bounds
    roll_targets = [np.random.uniform(roll_min, roll_max)]
    pitch_targets = [np.random.uniform(pitch_min, pitch_max)]
    
    # Generate progressive changes for remaining intervals
    for i in range(num_intervals - 1):
        # Random delta for roll
        delta_roll = np.random.uniform(-max_delta_roll, max_delta_roll)
        new_roll = np.clip(roll_targets[-1] + delta_roll, roll_min, roll_max)
        roll_targets.append(new_roll)
        
        # Random delta for pitch
        delta_pitch = np.random.uniform(-max_delta_pitch, max_delta_pitch)
        new_pitch = np.clip(pitch_targets[-1] + delta_pitch, pitch_min, pitch_max)
        pitch_targets.append(new_pitch)
    
    return np.array(roll_targets), np.array(pitch_targets)


def expand_targets_to_trajectory(roll_targets_array, pitch_targets_array,
                                 target_change_interval=100):
    """
    Expand interval-based targets to full trajectory length.
    
    Args:
        roll_targets_array: Array of target roll angles (one per interval)
        pitch_targets_array: Array of target pitch angles (one per interval)
        target_change_interval: Steps per interval
    
    Returns:
        Tuple of (roll_targets_full, pitch_targets_full) arrays of length NUM_STEPS
    """
    roll_targets_full = []
    pitch_targets_full = []
    
    for interval_idx in range(len(roll_targets_array)):
        roll_targets_full.extend([roll_targets_array[interval_idx]] * target_change_interval)
        pitch_targets_full.extend([pitch_targets_array[interval_idx]] * target_change_interval)
    
    return np.array(roll_targets_full), np.array(pitch_targets_full)


def run_single_trajectory(env, trajectory_num):
    """
    Run a single attitude control trajectory with convergence-triggered target changes.

    A new target is only set once BOTH roll AND pitch have been within
    CONVERGENCE_TOL_DEG for CONVERGENCE_STEPS consecutive steps, ensuring the
    dataset contains near-settled transitions at each target before moving on.

    Returns:
        Dictionary with trajectory metadata and list of transitions.
    """
    # Dynamic target initialisation — starts at a random point within bounds
    current_roll_deg  = round(np.random.uniform(ROLL_MIN, ROLL_MAX))
    current_pitch_deg = round(np.random.uniform(PITCH_MIN, PITCH_MAX))
    consec_in_tol     = 0          # consecutive steps both channels within tolerance
    num_target_changes = 0
    targets_log = [(current_roll_deg, current_pitch_deg)]  # for metadata
    
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
    
    # Run trajectory and collect transitions
    episode_reward = 0.0
    transitions = []  # List of (s_t, a_t, s_t+1, r_t, terminal_flag, target_roll, target_pitch)
    roll_angles = []
    pitch_angles = []
    roll_errors = []
    pitch_errors = []
    
    for step in range(NUM_STEPS):
        target_roll_current  = np.deg2rad(current_roll_deg)
        target_pitch_current = np.deg2rad(current_pitch_deg)

        # Store current state (s_t)
        state_t = obs.copy()

        # Extract state for PID
        roll    = obs[0]
        pitch   = obs[1]
        p_radps = obs[3]
        q_radps = obs[4]

        # Update PID references and compute commands
        pid_aileron.set_reference(target_roll_current)
        pid_elevator.set_reference(target_pitch_current)

        aileron_cmd,  _, _ = pid_aileron.update(state=roll,  state_dot=p_radps, saturate=True, normalize=False)
        elevator_cmd, _, _ = pid_elevator.update(state=pitch, state_dot=q_radps, saturate=True, normalize=False)

        # Persistent excitation: add Gaussian noise to PID outputs before applying.
        # The noise is model-independent, covering the wider action range used at MPPI
        # inference while the PID maintains trajectory stability.
        aileron_cmd  = np.clip(aileron_cmd  + np.random.normal(0.0, EXCITATION_NOISE_STD), -1.0, 1.0)
        elevator_cmd = np.clip(elevator_cmd + np.random.normal(0.0, EXCITATION_NOISE_STD), -1.0, 1.0)

        # Store action (a_t) — throttle filled in after step
        action = np.array([aileron_cmd, elevator_cmd, 0.0])

        obs, reward, terminated, truncated, info = env.step(action[:2])
        episode_reward += reward

        throttle   = env.unwrapped.sim[prp.throttle_cmd]
        action[2]  = throttle

        state_next = obs.copy()
        terminal   = terminated or truncated

        # Convergence-triggered target change: switch only after both channels
        # have been within CONVERGENCE_TOL_DEG for CONVERGENCE_STEPS steps.
        roll_err_deg  = abs(np.rad2deg(obs[0]) - current_roll_deg)
        pitch_err_deg = abs(np.rad2deg(obs[1]) - current_pitch_deg)
        if roll_err_deg < CONVERGENCE_TOL_DEG and pitch_err_deg < CONVERGENCE_TOL_DEG:
            consec_in_tol += 1
        else:
            consec_in_tol = 0
        if consec_in_tol >= CONVERGENCE_STEPS:
            current_roll_deg  = round(np.clip(
                current_roll_deg  + np.random.uniform(-MAX_DELTA_ROLL,  MAX_DELTA_ROLL),
                ROLL_MIN, ROLL_MAX
            ))
            current_pitch_deg = round(np.clip(
                current_pitch_deg + np.random.uniform(-MAX_DELTA_PITCH, MAX_DELTA_PITCH),
                PITCH_MIN, PITCH_MAX
            ))
            consec_in_tol = 0
            num_target_changes += 1
            targets_log.append((current_roll_deg, current_pitch_deg))

        transitions.append({
            'state_t':      state_t,
            'action':       action,
            'state_next':   state_next,
            'reward':       reward,
            'terminal':     terminal,
            'target_roll':  current_roll_deg,
            'target_pitch': current_pitch_deg,
        })

        roll_angles.append(np.rad2deg(obs[0]))
        pitch_angles.append(np.rad2deg(obs[1]))

        if len(obs) > 7:
            roll_errors.append(np.rad2deg(obs[6]))
            pitch_errors.append(np.rad2deg(obs[7]))
        else:
            roll_errors.append(np.rad2deg(target_roll_current  - obs[0]))
            pitch_errors.append(np.rad2deg(target_pitch_current - obs[1]))

        if terminal:
            break

    env.close()

    avg_roll        = np.mean(roll_angles)        if roll_angles  else 0.0
    avg_pitch       = np.mean(pitch_angles)       if pitch_angles else 0.0
    avg_roll_error  = np.mean(np.abs(roll_errors))  if roll_errors  else 0.0
    avg_pitch_error = np.mean(np.abs(pitch_errors)) if pitch_errors else 0.0

    trajectory_data = {
        'metadata': {
            'trajectory_num':    trajectory_num,
            'roll_targets':      np.array([t[0] for t in targets_log]),
            'pitch_targets':     np.array([t[1] for t in targets_log]),
            'avg_roll':          avg_roll,
            'avg_pitch':         avg_pitch,
            'avg_roll_error':    avg_roll_error,
            'avg_pitch_error':   avg_pitch_error,
            'total_reward':      episode_reward,
            'steps_executed':    len(roll_angles),
            'num_target_changes': num_target_changes,
        },
        'transitions': transitions,
    }

    print(f"[Traj {trajectory_num:2d}] Avg Roll={avg_roll:6.2f}°, Avg Pitch={avg_pitch:6.2f}°, "
          f"Errors: Roll={avg_roll_error:5.2f}°, Pitch={avg_pitch_error:5.2f}° | "
          f"Target changes: {num_target_changes} | Transitions: {len(transitions)}")
    
    return trajectory_data


def save_trajectory_data_to_csv(trajectory_results, output_file='updated_trajectory_data_progressive_noatmo_2.0.csv'):
    """
    Save trajectory data (state transitions) to CSV file for dynamics learning.
    
    CSV format:
        trajectory_id, step_id, target_roll, target_pitch,
        s_t_0, s_t_1, ..., s_t_13 (state at time t),
        a_t_0, a_t_1, a_t_2 (action: aileron, elevator, throttle),
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
            
            for step_id, transition in enumerate(traj_data['transitions']):
                row = {
                    'trajectory_id': trajectory_id,
                    'step_id': step_id,
                    'target_roll': transition['target_roll'],
                    'target_pitch': transition['target_pitch']
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
    print(f"  CSV format: trajectory_id, step_id, target_roll, target_pitch,")
    print(f"              s_t_0-13 (state), a_t_0-2 (action), s_t+1_0-13 (next state), reward, terminal")
    return output_file


def print_trajectory_summary(trajectory_results):
    """Print a formatted table of trajectory results."""
    print("\n" + "="*130)
    print("TRAJECTORY SUMMARY")
    print("="*130)
    print(f"{'Traj':>4} | {'Avg Roll':>10} | {'Avg Pitch':>11} | {'Roll Error':>11} | {'Pitch Error':>12} | "
          f"{'Reward':>10} | {'Steps':>6} | {'Transitions':>12}")
    print("-" * 130)
    
    for traj in trajectory_results:
        meta = traj['metadata']
        num_transitions = len(traj['transitions'])
        
        print(f"{meta['trajectory_num']:4d} | {meta['avg_roll']:10.2f}° | {meta['avg_pitch']:11.2f}° | "
              f"{meta['avg_roll_error']:11.2f}° | {meta['avg_pitch_error']:12.2f}° | "
              f"{meta['total_reward']:10.4f} | {meta['steps_executed']:6d} | {num_transitions:12d}")
    
    print("-" * 130)
    
    # Compute and print averages
    num_trajectories = len(trajectory_results)
    total_transitions = sum(len(t['transitions']) for t in trajectory_results)
    avg_obs_roll = np.mean([t['metadata']['avg_roll'] for t in trajectory_results])
    avg_obs_pitch = np.mean([t['metadata']['avg_pitch'] for t in trajectory_results])
    avg_roll_err = np.mean([t['metadata']['avg_roll_error'] for t in trajectory_results])
    avg_pitch_err = np.mean([t['metadata']['avg_pitch_error'] for t in trajectory_results])
    avg_reward = np.mean([t['metadata']['total_reward'] for t in trajectory_results])
    
    print(f"{'AVG':>4} | {avg_obs_roll:10.2f}° | {avg_obs_pitch:11.2f}° | "
          f"{avg_roll_err:11.2f}° | {avg_pitch_err:12.2f}° | "
          f"{avg_reward:10.4f} |")
    print("="*130 + "\n")
    
    print(f"Total trajectories generated: {num_trajectories}")
    print(f"Total state transitions collected: {total_transitions}")
    print(f"Configuration:")
    print(f"  - Trajectory length: {NUM_STEPS} steps (20 seconds at 100 Hz)")
    print(f"  - Target change: convergence-triggered (tol={CONVERGENCE_TOL_DEG}°, {CONVERGENCE_STEPS} steps)")
    print(f"  - Roll bounds: [{ROLL_MIN}, {ROLL_MAX}]°")
    print(f"  - Pitch bounds: [{PITCH_MIN}, {PITCH_MAX}]°")
    print(f"  - Max delta per interval: {MAX_DELTA_ROLL}° (roll), {MAX_DELTA_PITCH}° (pitch)\n")


def generate_progressive_trajectories(cfg: DictConfig, jsbsim_config_name='noatmo', 
                                       num_trajectories=NUM_TRAJECTORIES):
    """
    Generate trajectories with progressive target angles.
    
    Args:
        cfg: Hydra configuration
        jsbsim_config_name: Name of the JSBSim config to use
        num_trajectories: Number of trajectories to generate
    
    Returns:
        List of trajectory result dictionaries
    """
    trajectory_results = []
    
    print("\n" + "="*120)
    print(f"GENERATING PROGRESSIVE TRAJECTORIES FOR ENVIRONMENT: {jsbsim_config_name.upper()}")
    print("="*120)
    print(f"JSBSim config: {jsbsim_config_name}")
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Trajectory length: {NUM_STEPS} steps")
    print(f"Target change: convergence-triggered (tol={CONVERGENCE_TOL_DEG}°, {CONVERGENCE_STEPS} steps)")
    print(f"Roll bounds: [{ROLL_MIN}, {ROLL_MAX}]°, Pitch bounds: [{PITCH_MIN}, {PITCH_MAX}]°")
    print("="*120 + "\n")
    
    # Generate trajectories
    for traj_num in range(1, num_trajectories + 1):
        print(f"Generating trajectory {traj_num}/{num_trajectories}...")
        
        # Load configuration
        cfg.env.jsbsim = OmegaConf.load(f'../../config/env/jsbsim/{jsbsim_config_name}.yaml')

        try:
            # Create fresh environment for each trajectory
            env = gym.make(
                'ACBohnNoVaIErr-v0',
                cfg_env=cfg.env,
                render_mode='none'
            )

            # Run trajectory (targets generated dynamically inside)
            trajectory_data = run_single_trajectory(env, traj_num)
            
            trajectory_results.append(trajectory_data)
            
        except Exception as e:
            print(f"Error in trajectory {traj_num}: {e}")
            import traceback
            traceback.print_exc()
    
    return trajectory_results


def main(cfg: DictConfig):
    """Main function to generate progressive trajectories and save to CSV."""
    
    try:
        # Generate trajectories for the configuration
        print(f"\n{'='*100}")
        print(f"Processing environment configuration: noatmo")
        print(f"{'='*100}")
        
        trajectory_results = generate_progressive_trajectories(
            cfg, 
            jsbsim_config_name='noatmo',
            num_trajectories=NUM_TRAJECTORIES
        )
        
        # Print summary table
        print_trajectory_summary(trajectory_results)
        
        # Save trajectory data to CSV file
        output_file = str(Path(__file__).parent / 'trajectory_data_pid_converged.csv')
        csv_file = save_trajectory_data_to_csv(trajectory_results, output_file)
        
        print(f"\n{'='*100}")
        print("✓ Data collection with progressive targets complete!")
        print(f"{'='*100}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_name='default', config_path='../../config', version_base=None)
def app(cfg: DictConfig):
    main(cfg)


if __name__ == '__main__':
    app()
