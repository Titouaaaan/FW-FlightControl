"""
Trajectory Data Collection Script
Generates trajectories systematically to cover the observable space.
Saves trajectory statistics without file logging.

Usage:
    python data_collection.py
"""

import numpy as np
import gymnasium as gym
import fw_jsbgym
import hydra
from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from omegaconf import DictConfig
import itertools


# ============================================================================
# CONFIGURATION
# ============================================================================
# Define systematic target angles for space coverage (in degrees)
ROLL_TARGETS = [5, 10, 15, 20, 25, 30, 35]      # 7 roll angles
PITCH_TARGETS = [5, 15, 20]                      # 3 pitch angles
# Total trajectories: 7 * 3 = 21 (easily extensible by adding more values)

NUM_STEPS = 2000  # 20 seconds at 100 Hz
TRAJECTORIES = []  # Will store trajectory history for future incremental additions


def run_single_trajectory(env, trajectory_num, target_roll_deg, target_pitch_deg):
    """
    Run a single attitude control trajectory with PID controllers.
    
    Args:
        env: The attitude control environment
        trajectory_num: Trajectory number for display
        target_roll_deg: Target roll angle in degrees
        target_pitch_deg: Target pitch angle in degrees
    
    Returns:
        Dictionary with trajectory statistics:
        - target_roll, target_pitch
        - avg_roll, avg_pitch (actual observed angles)
        - avg_roll_error, avg_pitch_error
        - total_reward
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
    
    # Run trajectory
    episode_reward = 0.0
    roll_angles = []
    pitch_angles = []
    roll_errors = []
    pitch_errors = []
    
    for step in range(NUM_STEPS):
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
        
        action = np.array([aileron_cmd, elevator_cmd])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Track angles and errors
        roll_angles.append(np.rad2deg(obs[0]))
        pitch_angles.append(np.rad2deg(obs[1]))
        
        if len(obs) > 7:
            roll_errors.append(np.rad2deg(obs[6]))
            pitch_errors.append(np.rad2deg(obs[7]))
        else:
            roll_errors.append(np.rad2deg(target_roll_rad - obs[0]))
            pitch_errors.append(np.rad2deg(target_pitch_rad - obs[1]))
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Compute statistics
    avg_roll = np.mean(roll_angles) if roll_angles else 0.0
    avg_pitch = np.mean(pitch_angles) if pitch_angles else 0.0
    avg_roll_error = np.mean(np.abs(roll_errors)) if roll_errors else 0.0
    avg_pitch_error = np.mean(np.abs(pitch_errors)) if pitch_errors else 0.0
    
    trajectory_data = {
        'trajectory_num': trajectory_num,
        'target_roll': target_roll_deg,
        'target_pitch': target_pitch_deg,
        'avg_roll': avg_roll,
        'avg_pitch': avg_pitch,
        'avg_roll_error': avg_roll_error,
        'avg_pitch_error': avg_pitch_error,
        'total_reward': episode_reward,
        'steps_executed': len(roll_angles)
    }
    
    print(f"[Traj {trajectory_num:2d}] Targets: Roll={target_roll_deg:5.1f}°, Pitch={target_pitch_deg:5.1f}° → "
          f"Avg Roll={avg_roll:6.2f}°, Avg Pitch={avg_pitch:6.2f}°, "
          f"Errors: Roll={avg_roll_error:5.2f}°, Pitch={avg_pitch_error:5.2f}°")
    
    return trajectory_data


def print_trajectory_summary(trajectory_results):
    """Print a formatted table of trajectory results."""
    print("\n" + "="*120)
    print("TRAJECTORY SUMMARY")
    print("="*120)
    print(f"{'Traj':>4} | {'Target Roll':>12} | {'Target Pitch':>13} | {'Avg Roll':>10} | {'Avg Pitch':>11} | "
          f"{'Roll Error':>11} | {'Pitch Error':>12} | {'Reward':>10} | {'Steps':>6}")
    print("-" * 120)
    
    for traj in trajectory_results:
        print(f"{traj['trajectory_num']:4d} | {traj['target_roll']:12.2f}° | {traj['target_pitch']:13.2f}° | "
              f"{traj['avg_roll']:10.2f}° | {traj['avg_pitch']:11.2f}° | "
              f"{traj['avg_roll_error']:11.2f}° | {traj['avg_pitch_error']:12.2f}° | "
              f"{traj['total_reward']:10.4f} | {traj['steps_executed']:6d}")
    
    print("-" * 120)
    
    # Compute and print averages
    num_trajectories = len(trajectory_results)
    avg_target_roll = np.mean([t['target_roll'] for t in trajectory_results])
    avg_target_pitch = np.mean([t['target_pitch'] for t in trajectory_results])
    avg_obs_roll = np.mean([t['avg_roll'] for t in trajectory_results])
    avg_obs_pitch = np.mean([t['avg_pitch'] for t in trajectory_results])
    avg_roll_err = np.mean([t['avg_roll_error'] for t in trajectory_results])
    avg_pitch_err = np.mean([t['avg_pitch_error'] for t in trajectory_results])
    avg_reward = np.mean([t['total_reward'] for t in trajectory_results])
    
    print(f"{'AVG':>4} | {avg_target_roll:12.2f}° | {avg_target_pitch:13.2f}° | "
          f"{avg_obs_roll:10.2f}° | {avg_obs_pitch:11.2f}° | "
          f"{avg_roll_err:11.2f}° | {avg_pitch_err:12.2f}° | "
          f"{avg_reward:10.4f} |")
    print("="*120 + "\n")
    
    print(f"Total trajectories generated: {num_trajectories}")
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
    """Main function to generate systematic trajectories."""
    
    try:
        # Generate trajectories with systematic coverage
        trajectory_results = generate_systematic_trajectories(cfg)
        
        # Print summary table
        print_trajectory_summary(trajectory_results)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_name='default', config_path='config', version_base=None)
def app(cfg: DictConfig):
    main(cfg)


if __name__ == '__main__':
    app()
