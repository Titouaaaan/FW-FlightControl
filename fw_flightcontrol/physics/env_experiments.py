"""
Environment Experiments Script with PID Controllers
Test attitude control with different environmental conditions (wind, gusts, turbulence).
Runs the attitude control environment and controls roll, pitch using PID controllers.

Usage:
    python env_experiments.py
    python env_experiments.py roll_limit=45 pitch_limit=25

Available JSBSim Configs (change JSBSIM_CONFIG below):
    - 'noatmo': No wind, turbulence, or gusts
    - 'constwind': Constant light wind from north
    - 'gustsonly': Moderate gusts only
    - 'turbonly': Moderate turbulence only
    - 'alldist': All disturbances (wind + turbulence + gusts)
"""

import numpy as np
import gymnasium as gym
import fw_jsbgym
import hydra
from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.models.aerodynamics import AeroModel
from fw_jsbgym.utils import jsbsim_properties as prp
from omegaconf import DictConfig, OmegaConf

# ============================================================================
# CONFIGURATION: Change the JSBSim environment here
# ============================================================================
# Select which JSBSim configuration to use:
#   - 'noatmo': No atmospheric disturbances (baseline)
#   - 'constwind': Constant wind from the north
#   - 'gustsonly': Gusts only (no wind/turbulence)
#   - 'turbonly': Turbulence only
#   - 'alldist': All disturbances combined
JSBSIM_CONFIG = 'constwind'  
# ============================================================================


def print_env_info(env):
    """Print comprehensive information about the environment."""
    print("\n" + "="*80)
    print("ENVIRONMENT INFORMATION")
    print("="*80)
    
    # Basic environment info
    print(f"\nEnvironment ID: {env.spec.id if env.spec else 'Unknown'}")
    print(f"Render Mode: {env.unwrapped.render_mode}")
    print(f"Max Episode Steps: {env.unwrapped.max_episode_steps}")
    print(f"Episode Length (s): {env.unwrapped.episode_length_s}")
    print(f"Agent Frequency (Hz): {env.unwrapped.agent_frequency}")
    print(f"FDM Frequency (Hz): {env.unwrapped.fdm_frequency}")
    print(f"FDM dt (s): {env.unwrapped.fdm_dt}")
    
    # Action space
    print(f"\n--- ACTION SPACE ---")
    print(f"Action Space Shape: {env.action_space.shape}")
    print(f"Action Space Low:  {env.action_space.low}")
    print(f"Action Space High: {env.action_space.high}")
    print(f"Number of Actions: {len(env.unwrapped.action_prps)}")
    print("Action Variables:")
    for i, prop in enumerate(env.unwrapped.action_prps):
        print(f"  [{i}] {prop.get_legal_name():30s} - bounds: [{prop.min:8.3f}, {prop.max:8.3f}]")
    
    # State/Observation space
    print(f"\n--- STATE/OBSERVATION SPACE ---")
    print(f"Observation Space Shape: {env.observation_space.shape}")
    print(f"Observation Space Low:  {env.observation_space.low}")
    print(f"Observation Space High: {env.observation_space.high}")
    print(f"Number of State Variables: {len(env.unwrapped.state_prps)}")
    print("State Variables:")
    for i, prop in enumerate(env.unwrapped.state_prps):
        print(f"  [{i}] {prop.get_legal_name():30s} - bounds: [{prop.min:8.3f}, {prop.max:8.3f}]")
    
    # Target space
    print(f"\n--- TARGET STATE SPACE ---")
    print(f"Number of Target Variables: {len(env.unwrapped.target_prps)}")
    print("Target Variables:")
    for i, prop in enumerate(env.unwrapped.target_prps):
        print(f"  [{i}] {prop.get_legal_name():30s} - bounds: [{prop.min:8.3f}, {prop.max:8.3f}]")
    
    # Error space
    print(f"\n--- ERROR SPACE ---")
    print(f"Number of Error Variables: {len(env.unwrapped.error_prps)}")
    print("Error Variables:")
    for i, prop in enumerate(env.unwrapped.error_prps):
        print(f"  [{i}] {prop.get_legal_name():30s}")
    
    print("\n" + "="*80)


def print_state_info(obs, env):
    """Print current state information."""
    print(f"\n--- Current State (Observation) ---")
    for i, prop in enumerate(env.unwrapped.state_prps):
        if i < len(obs):
            print(f"  [{i}] {prop.get_legal_name():30s} = {obs[i]:10.5f}")


def print_step_info(step, obs, env, target_roll, target_pitch, reward):
    """Print step information."""
    # Extract current attitude and errors
    roll_idx = 0  # roll is first in state_prps
    pitch_idx = 1  # pitch is second
    airspeed_idx = 2  # airspeed is third
    p_idx = 3  # roll rate
    q_idx = 4  # pitch rate
    r_idx = 5  # yaw rate
    
    # Error indices depend on configuration, but typically after rates
    if len(obs) > 10:  # full state space
        roll_err_idx = 6
        pitch_err_idx = 7
    else:
        roll_err_idx = 6
        pitch_err_idx = 7
    
    current_roll = np.rad2deg(obs[roll_idx])
    current_pitch = np.rad2deg(obs[pitch_idx])
    current_airspeed = obs[airspeed_idx]
    current_p = np.rad2deg(obs[p_idx])
    current_q = np.rad2deg(obs[q_idx])
    current_r = np.rad2deg(obs[r_idx])
    
    roll_error = np.rad2deg(obs[roll_err_idx])
    pitch_error = np.rad2deg(obs[pitch_err_idx])
    
    if step % 100 == 0:
        print(f"\nStep {step:5d} | "
              f"Roll: {current_roll:6.2f}° (target: {np.rad2deg(target_roll):6.2f}°) err: {roll_error:6.2f}° | "
              f"Pitch: {current_pitch:6.2f}° (target: {np.rad2deg(target_pitch):6.2f}°) err: {pitch_error:6.2f}° | "
              f"Va: {current_airspeed:6.2f} kph | Reward: {reward:7.4f}")


def run_attitude_control_loop(env, num_steps=2000, target_roll_deg=20, target_pitch_deg=10):
    """
    Run the attitude control loop with PID controllers.
    
    Args:
        env: The attitude control environment
        num_steps: Number of simulation steps
        target_roll_deg: Target roll angle in degrees
        target_pitch_deg: Target pitch angle in degrees
    """
    
    # Convert targets to radians
    target_roll_rad = np.deg2rad(target_roll_deg)
    target_pitch_rad = np.deg2rad(target_pitch_deg)
    
    print(f"\n{'='*80}")
    print(f"ATTITUDE CONTROL LOOP")
    print(f"{'='*80}")
    print(f"Target Roll:  {target_roll_deg:.1f}°")
    print(f"Target Pitch: {target_pitch_deg:.1f}°")
    print(f"Number of Steps: {num_steps}")
    print(f"Simulation Duration: {num_steps * env.unwrapped.fdm_dt:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Initialize PID controllers
    # Aileron controller for roll control
    # Tuned for X8 fixed-wing UAV
    pid_aileron = PID(
        kp=1.5,     # proportional gain - reduce for gentler response
        ki=0.1,     # integral gain - reduce to prevent integral windup
        kd=0.1,     # derivative gain - provides damping
        dt=env.unwrapped.fdm_dt,
        trim=TrimPoint(),
        limit=1.0,
        is_throttle=False
    )
    
    # Elevator controller for pitch control
    # More carefully tuned for pitch since it's more unstable
    pid_elevator = PID(
        kp=-2.0,     # proportional gain - more conservative
        ki=-0.3,     # integral gain - very conservative
        kd=-0.1,     # derivative gain - higher for pitch damping
        dt=env.unwrapped.fdm_dt,
        trim=TrimPoint(),
        limit=1.0,
        is_throttle=False
    )
    
    print("PID Controller Gains:")
    print(f"  Aileron (Roll):   Kp={pid_aileron.kp:.3f}, Ki={pid_aileron.ki:.3f}, Kd={pid_aileron.kd:.3f}")
    print(f"  Elevator (Pitch): Kp={pid_elevator.kp:.3f}, Ki={pid_elevator.ki:.3f}, Kd={pid_elevator.kd:.3f}")
    print()
    
    # Initialize the environment (must be called before reset)
    print("Initializing environment...")
    env.unwrapped.init()
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset. Initial observation shape: {obs.shape}")
    
    # Set target state (roll, pitch)
    target_state = np.array([target_roll_rad, target_pitch_rad])
    env.set_target_state(target_state)
    
    # Print environment information
    print_env_info(env)
    
    # Initialize statistics
    episode_reward = 0.0
    rewards_list = []
    roll_errors = []
    pitch_errors = []
    
    # Simulation loop
    print(f"\nStarting simulation loop...")
    print("-" * 80)
    
    try:
        for step in range(num_steps):
            # Extract state for PID (assuming standard state order)
            # State order: [roll, pitch, airspeed, p, q, r, roll_err, pitch_err, ...]
            if len(obs) > 6:
                roll = obs[0]
                pitch = obs[1]
                p_radps = obs[3]  # roll rate
                q_radps = obs[4]  # pitch rate
                
                roll_err = obs[6] if len(obs) > 6 else (target_roll_rad - roll)
                pitch_err = obs[7] if len(obs) > 7 else (target_pitch_rad - pitch)
            else:
                roll = obs[0]
                pitch = obs[1]
                p_radps = obs[3]
                q_radps = obs[4]
                roll_err = target_roll_rad - roll
                pitch_err = target_pitch_rad - pitch
            
            # Update target and PID references
            pid_aileron.set_reference(target_roll_rad)
            pid_elevator.set_reference(target_pitch_rad)
            
            # Compute PID commands with derivative term (use angular rates)
            aileron_cmd, _, _ = pid_aileron.update(state=roll, state_dot=p_radps, saturate=True, normalize=False)
            elevator_cmd, _, _ = pid_elevator.update(state=pitch, state_dot=q_radps, saturate=True, normalize=False)
            
            # Action is [aileron, elevator] (no throttle, it's managed by environment)
            action = np.array([aileron_cmd, elevator_cmd])
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            rewards_list.append(reward)
            
            # Track errors
            if len(obs) > 7:
                roll_errors.append(np.rad2deg(obs[6]))
                pitch_errors.append(np.rad2deg(obs[7]))
            
            # Print step info periodically
            print_step_info(step, obs, env, target_roll_rad, target_pitch_rad, reward)
            
            # Check termination
            if terminated or truncated:
                print(f"\nEpisode terminated/truncated at step {step}")
                break
        
        # Print final statistics
        print("\n" + "="*80)
        print("SIMULATION COMPLETED")
        print("="*80)
        print(f"Total Steps Executed: {step + 1}")
        print(f"Total Episode Reward: {episode_reward:.4f}")
        print(f"Mean Step Reward: {np.mean(rewards_list):.6f}")
        print(f"Std Step Reward: {np.std(rewards_list):.6f}")
        
        if roll_errors:
            roll_errors = np.array(roll_errors)
            pitch_errors = np.array(pitch_errors)
            print(f"\nRoll Tracking:")
            print(f"  Mean Error: {np.mean(np.abs(roll_errors)):.4f}°")
            print(f"  RMSE: {np.sqrt(np.mean(roll_errors**2)):.4f}°")
            print(f"  Max Error: {np.max(np.abs(roll_errors)):.4f}°")
            
            print(f"\nPitch Tracking:")
            print(f"  Mean Error: {np.mean(np.abs(pitch_errors)):.4f}°")
            print(f"  RMSE: {np.sqrt(np.mean(pitch_errors**2)):.4f}°")
            print(f"  Max Error: {np.max(np.abs(pitch_errors)):.4f}°")
        
        print(f"\nFinal State:")
        print_state_info(obs, env)
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\nEnvironment closed.")


def main(cfg: DictConfig):
    """Main function to run environment experiments with attitude control."""
    
    print("\n" + "="*80)
    print(f"ATTITUDE CONTROL ENVIRONMENT EXPERIMENTS - {JSBSIM_CONFIG.upper()}")
    print("="*80 + "\n")
    
    try:
        # Load the configured JSBSim settings
        print(f"Loading JSBSim configuration: '{JSBSIM_CONFIG}'")
        cfg.env.jsbsim = OmegaConf.load(f'config/env/jsbsim/{JSBSIM_CONFIG}.yaml')
        
        # Create the environment using gymnasium
        print("Creating environment 'ACBohnNoVaIErr-v0'...")
        env = gym.make(
            'ACBohnNoVaIErr-v0',
            cfg_env=cfg.env,
            telemetry_file='telemetry/attitude_control_telemetry.csv',
            render_mode='none'
        )
        
        print(f"Environment created successfully!")
        print(f"Environment type: {type(env)}")
        
        # Run the attitude control loop with target angles from config
        target_roll = cfg.roll_limit
        target_pitch = cfg.pitch_limit

        print(f"\nConfiguration:")
        print(f"  JSBSim Config: {JSBSIM_CONFIG}")
        print(f"  Roll Limit:    {target_roll}°")
        print(f"  Pitch Limit:   {target_pitch}°")
        
        run_attitude_control_loop(
            env,
            num_steps=2000,
            target_roll_deg=target_roll,
            target_pitch_deg=target_pitch
        )
        
    except Exception as e:
        print(f"\nError in main: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(config_name='default', config_path='config', version_base=None)
def app(cfg: DictConfig):
    main(cfg)


if __name__ == '__main__':
    app()
