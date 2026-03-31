"""
Attitude Control Data Collection Script
Runs N random attitude control experiments and prints results to terminal.

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


# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_EXPERIMENTS = 5
NUM_STEPS = 2000  # 20 seconds at 100 Hz


def run_single_experiment(env, experiment_num, target_roll_deg, target_pitch_deg):
    """
    Run a single attitude control experiment with PID controllers.
    
    Args:
        env: The attitude control environment
        experiment_num: Experiment number for display
        target_roll_deg: Target roll angle in degrees
        target_pitch_deg: Target pitch angle in degrees
    
    Returns:
        Tuple of (roll_rmse, pitch_rmse, total_reward)
    """
    
    # Convert targets to radians
    target_roll_rad = np.deg2rad(target_roll_deg)
    target_pitch_rad = np.deg2rad(target_pitch_deg)
    
    print(f"\n[Exp {experiment_num}/{NUM_EXPERIMENTS}] Target: Roll={target_roll_deg:.1f}°, Pitch={target_pitch_deg:.1f}°")
    
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
    
    # Run simulation
    episode_reward = 0.0
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
        
        # Track errors
        if len(obs) > 7:
            roll_errors.append(np.rad2deg(obs[6]))
            pitch_errors.append(np.rad2deg(obs[7]))
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Compute statistics
    roll_errors = np.array(roll_errors)
    pitch_errors = np.array(pitch_errors)
    roll_rmse = np.sqrt(np.mean(roll_errors**2)) if len(roll_errors) > 0 else 0.0
    pitch_rmse = np.sqrt(np.mean(pitch_errors**2)) if len(pitch_errors) > 0 else 0.0
    
    print(f"  → Roll RMSE: {roll_rmse:.4f}°, Pitch RMSE: {pitch_rmse:.4f}°, Reward: {episode_reward:.4f}")
    
    return roll_rmse, pitch_rmse, episode_reward


def main(cfg: DictConfig):
    """Main function to run N random attitude control experiments."""
    
    print("\n" + "="*80)
    print(f"RUNNING {NUM_EXPERIMENTS} RANDOM ATTITUDE CONTROL EXPERIMENTS")
    print("="*80)
    print(f"Target angles: Roll and Pitch each in range [10°, 20°]")
    print(f"Steps per experiment: {NUM_STEPS}")
    print("="*80)
    
    results = []
    
    try:
        for exp_num in range(1, NUM_EXPERIMENTS + 1):
            # Generate random target angles between 10 and 20 degrees
            target_roll = np.random.uniform(10, 20)
            target_pitch = np.random.uniform(10, 20)
            
            # Create fresh environment for each experiment
            env = gym.make(
                'ACBohnNoVaIErr-v0',
                cfg_env=cfg.env,
                render_mode='none'
            )
            
            # Run experiment
            roll_rmse, pitch_rmse, reward = run_single_experiment(
                env, exp_num, target_roll, target_pitch
            )
            
            results.append({
                'exp': exp_num,
                'target_roll': target_roll,
                'target_pitch': target_pitch,
                'roll_rmse': roll_rmse,
                'pitch_rmse': pitch_rmse,
                'reward': reward
            })
        
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Exp':>3} | {'Target Roll':>12} | {'Target Pitch':>13} | {'Roll RMSE':>10} | {'Pitch RMSE':>11} | {'Reward':>10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['exp']:3d} | {r['target_roll']:12.2f}° | {r['target_pitch']:13.2f}° | {r['roll_rmse']:10.4f}° | {r['pitch_rmse']:11.4f}° | {r['reward']:10.4f}")
        
        print("-" * 80)
        roll_rmses = np.array([r['roll_rmse'] for r in results])
        pitch_rmses = np.array([r['pitch_rmse'] for r in results])
        rewards = np.array([r['reward'] for r in results])
        
        print(f"{'AVG':>3} | {np.mean([r['target_roll'] for r in results]):12.2f}° | {np.mean([r['target_pitch'] for r in results]):13.2f}° | {np.mean(roll_rmses):10.4f}° | {np.mean(pitch_rmses):11.4f}° | {np.mean(rewards):10.4f}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_name='default', config_path='config', version_base=None)
def app(cfg: DictConfig):
    main(cfg)


if __name__ == '__main__':
    app()
