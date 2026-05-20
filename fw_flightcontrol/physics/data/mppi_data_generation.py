#!/usr/bin/env python3
"""
MPPI (Prior-Only) Trajectory Data Collection

Generates training data using MPPI with the physics prior only (no residual).
This replaces the PID controller used in updated_data_collection.py with MPPI,
which produces more consistent, physically grounded trajectories — especially at
hard angles where the PID fails and contaminates the residual's training signal.

Scientific motivation:
    - The physics prior is a FIXED, known model: no circular dependency.
    - MPPI under the prior explores a wider, more uniform action distribution than
      the PID, removing the controller-induced bias from the residual's training data.
    - APHYNITY trains on real environment transitions regardless of what generated
      the actions, so the learned residual corrects true physics errors only.

The target schedule, trajectory length, CSV format, and number of trajectories are
identical to updated_data_collection.py for direct comparability.

Usage:
    cd fw_flightcontrol/physics/
    python mppi_data_generation.py
    python mppi_data_generation.py --num-trajectories 30 --num-steps 1000
    python mppi_data_generation.py --mppi-samples 600 --mppi-horizon 15  # full-quality
"""

import numpy as np
import gymnasium as gym
import fw_jsbgym
import sys
import csv
import argparse
import torch
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fw_flightcontrol.physics.mppi_test import (
    load_config_and_model,
    initialize_environment,
    MPPIController,
    rollout_trajectories,
)
from fw_jsbgym.utils import jsbsim_properties as prp


# ============================================================================
# CONFIGURATION  (mirrors updated_data_collection.py)
# ============================================================================
NUM_STEPS               = 1500
TARGET_CHANGE_INTERVAL_MIN = 80
TARGET_CHANGE_INTERVAL_MAX = 120

ROLL_MIN,  ROLL_MAX  = -60, 60
PITCH_MIN, PITCH_MAX = -30, 30
MAX_DELTA_ROLL  = 20
MAX_DELTA_PITCH = 15

NUM_TRAJECTORIES = 60

# MPPI parameters — balanced for collection speed vs. quality
MPPI_SAMPLES     = 300
MPPI_HORIZON     = 15
MPPI_TEMPERATURE = 1.0
MPPI_NOISE_STD   = 0.8

OUTPUT_CSV = str(
    Path(__file__).parent.parent / 'data' / 'trajectory_data_mppi_prior.csv'
)
DEFAULT_CHECKPOINT = str(
    Path(__file__).parent / 'checkpoints'
    / 'data_driven_normalization_3.0' / 'final_model.pt'
)
DEFAULT_CONFIG = str(Path(__file__).parent / 'training_params.yaml')


# ============================================================================
# SINGLE TRAJECTORY
# ============================================================================
def run_single_trajectory(
    env,
    trajectory_num: int,
    hybrid_model,
    model_config: dict,
    denorm_factors,
    min_bounds,
    norm_type,
    device: torch.device,
    mppi_samples: int,
    mppi_horizon: int,
    mppi_temperature: float,
    mppi_noise_std: float,
    num_steps: int,
) -> dict:
    """
    Run one data-collection trajectory with MPPI (prior-only) as the controller.

    Mirrors updated_data_collection.run_single_trajectory() but replaces the two
    PID controllers with a single MPPIController.  All recorded transitions are
    real environment transitions — the data reflects true physics, not the model.
    """
    # --- Dynamic target initialisation (same schedule as updated_data_collection) ---
    current_roll_deg  = np.random.uniform(ROLL_MIN,  ROLL_MAX)
    current_pitch_deg = np.random.uniform(PITCH_MIN, PITCH_MAX)
    steps_until_change = np.random.randint(
        TARGET_CHANGE_INTERVAL_MIN, TARGET_CHANGE_INTERVAL_MAX + 1
    )
    steps_in_current = 0
    targets_log = [(current_roll_deg, current_pitch_deg)]

    # --- MPPI controller ---
    controller = MPPIController(
        horizon=mppi_horizon,
        action_dim=3,
        num_samples=mppi_samples,
        temperature=mppi_temperature,
        noise_std=mppi_noise_std,
    )
    controller.reset()

    # --- Throttle override ---
    # env.step() accepts only [aileron, elevator]; the environment's internal PI
    # controller sets throttle.  We monkey-patch apply_action to overwrite the PI
    # throttle with MPPI's throttle immediately after the PI sets it (before the
    # physics step), replicating the same mechanism used in mppi_test.run_mppi_control.
    mppi_throttle_ref = [0.3]
    _original_apply = env.unwrapped.apply_action

    def _patched_apply_action(action):
        _original_apply(action)
        env.unwrapped.sim[prp.throttle_cmd] = float(
            np.clip(mppi_throttle_ref[0], 0.0, 1.0)
        )

    env.unwrapped.apply_action = _patched_apply_action

    # --- Environment reset ---
    env.unwrapped.init()
    obs, _ = env.reset()

    # --- Collection buffers ---
    transitions  = []
    roll_angles  = []
    pitch_angles = []
    roll_errors  = []
    pitch_errors = []
    episode_reward = 0.0

    pbar = tqdm(range(num_steps), desc=f"Traj {trajectory_num:2d}", unit="step", leave=False)
    for step in pbar:
        # Switch target when interval expires — same logic as updated_data_collection
        if steps_in_current >= steps_until_change:
            current_roll_deg = np.clip(
                current_roll_deg + np.random.uniform(-MAX_DELTA_ROLL, MAX_DELTA_ROLL),
                ROLL_MIN, ROLL_MAX,
            )
            current_pitch_deg = np.clip(
                current_pitch_deg + np.random.uniform(-MAX_DELTA_PITCH, MAX_DELTA_PITCH),
                PITCH_MIN, PITCH_MAX,
            )
            steps_until_change = np.random.randint(
                TARGET_CHANGE_INTERVAL_MIN, TARGET_CHANGE_INTERVAL_MAX + 1
            )
            steps_in_current = 0
            targets_log.append((current_roll_deg, current_pitch_deg))
            controller.reset()  # fresh warm-start for the new target

        steps_in_current += 1
        state_t = obs.copy()

        # MPPI optimisation (physics prior only — residual disabled at model level)
        best_action, _ = controller.optimize(
            obs,
            current_roll_deg,
            current_pitch_deg,
            hybrid_model,
            model_config,
            denorm_factors,
            min_bounds,
            norm_type,
            device,
        )

        # Apply: aileron + elevator via env, throttle via monkey-patch
        mppi_throttle_ref[0] = float(best_action[2])
        obs, reward, terminated, truncated, _ = env.step(best_action[:2])
        episode_reward += reward

        # Record the full 3-D MPPI action (including MPPI throttle, not PI throttle)
        action = np.array([
            float(best_action[0]),
            float(best_action[1]),
            float(best_action[2]),
        ])

        state_next = obs.copy()
        terminal   = terminated or truncated

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

        target_roll_rad  = np.deg2rad(current_roll_deg)
        target_pitch_rad = np.deg2rad(current_pitch_deg)
        roll_errors.append(abs(np.rad2deg(obs[0]) - current_roll_deg))
        pitch_errors.append(abs(np.rad2deg(obs[1]) - current_pitch_deg))
        pbar.set_postfix(roll_err=f"{roll_errors[-1]:.1f}°", pitch_err=f"{pitch_errors[-1]:.1f}°")

        if terminal:
            break

    # Restore original apply_action
    env.unwrapped.apply_action = _original_apply
    env.close()

    avg_roll        = float(np.mean(roll_angles))        if roll_angles  else 0.0
    avg_pitch       = float(np.mean(pitch_angles))       if pitch_angles else 0.0
    avg_roll_error  = float(np.mean(roll_errors))        if roll_errors  else 0.0
    avg_pitch_error = float(np.mean(pitch_errors))       if pitch_errors else 0.0

    metadata = {
        'trajectory_num':  trajectory_num,
        'roll_targets':    np.array([t[0] for t in targets_log]),
        'pitch_targets':   np.array([t[1] for t in targets_log]),
        'avg_roll':        avg_roll,
        'avg_pitch':       avg_pitch,
        'avg_roll_error':  avg_roll_error,
        'avg_pitch_error': avg_pitch_error,
        'total_reward':    episode_reward,
        'steps_executed':  len(transitions),
    }

    print(
        f"[Traj {trajectory_num:2d}] Avg Roll={avg_roll:6.2f}°  Avg Pitch={avg_pitch:6.2f}°  "
        f"Errors: Roll={avg_roll_error:5.2f}°  Pitch={avg_pitch_error:5.2f}°  "
        f"Steps: {len(transitions)}  Targets: {len(targets_log)}"
    )

    return {'metadata': metadata, 'transitions': transitions}


# ============================================================================
# CSV SAVING  (identical format to updated_data_collection.py)
# ============================================================================
def save_to_csv(trajectory_results: list, output_file: str) -> str:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['trajectory_id', 'step_id', 'target_roll', 'target_pitch']
    fieldnames += [f's_t_{i}'   for i in range(14)]
    fieldnames += [f'a_t_{i}'   for i in range(3)]
    fieldnames += [f's_t+1_{i}' for i in range(14)]
    fieldnames += ['reward', 'terminal']

    total = 0
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for traj_data in trajectory_results:
            traj_id = traj_data['metadata']['trajectory_num']
            for step_id, tr in enumerate(traj_data['transitions']):
                row = {
                    'trajectory_id': traj_id,
                    'step_id':       step_id,
                    'target_roll':   tr['target_roll'],
                    'target_pitch':  tr['target_pitch'],
                }
                for i in range(14):
                    row[f's_t_{i}']   = tr['state_t'][i]
                    row[f's_t+1_{i}'] = tr['state_next'][i]
                for i in range(3):
                    row[f'a_t_{i}'] = tr['action'][i]
                row['reward']   = tr['reward']
                row['terminal'] = 1 if tr['terminal'] else 0
                writer.writerow(row)
                total += 1

    print(f"\n✓ Saved {total} transitions to '{output_file}'")
    return output_file


# ============================================================================
# SUMMARY  (mirrors updated_data_collection.print_trajectory_summary)
# ============================================================================
def print_summary(trajectory_results: list):
    print("\n" + "=" * 130)
    print("TRAJECTORY SUMMARY")
    print("=" * 130)
    print(
        f"{'Traj':>4} | {'Avg Roll':>10} | {'Avg Pitch':>11} | "
        f"{'Roll Error':>11} | {'Pitch Error':>12} | {'Reward':>10} | {'Steps':>6}"
    )
    print("-" * 130)
    for t in trajectory_results:
        m = t['metadata']
        print(
            f"{m['trajectory_num']:4d} | {m['avg_roll']:10.2f}° | {m['avg_pitch']:11.2f}° | "
            f"{m['avg_roll_error']:11.2f}° | {m['avg_pitch_error']:12.2f}° | "
            f"{m['total_reward']:10.4f} | {m['steps_executed']:6d}"
        )
    total_steps = sum(t['metadata']['steps_executed'] for t in trajectory_results)
    print("-" * 130)
    print(f"Total trajectories: {len(trajectory_results)}")
    print(f"Total transitions:  {total_steps}")
    print(f"Config: steps={NUM_STEPS}, interval={TARGET_CHANGE_INTERVAL_MIN}–"
          f"{TARGET_CHANGE_INTERVAL_MAX}, roll=[{ROLL_MIN},{ROLL_MAX}]°, "
          f"pitch=[{PITCH_MIN},{PITCH_MAX}]°")
    print("=" * 130 + "\n")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Collect trajectory data using MPPI (physics prior only)'
    )
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Model checkpoint (residual will be disabled — only prior is used)')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help='Path to training_params.yaml')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                        help='Output CSV path')
    parser.add_argument('--num-trajectories', type=int, default=NUM_TRAJECTORIES)
    parser.add_argument('--num-steps',        type=int, default=NUM_STEPS,
                        help='Steps per trajectory (default 2000 = 20 s, matches PID dataset)')
    parser.add_argument('--mppi-samples',     type=int,   default=MPPI_SAMPLES)
    parser.add_argument('--mppi-horizon',     type=int,   default=MPPI_HORIZON)
    parser.add_argument('--mppi-temperature', type=float, default=MPPI_TEMPERATURE)
    parser.add_argument('--mppi-noise-std',   type=float, default=MPPI_NOISE_STD)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load model, then DISABLE the residual ---
    hybrid_model, model_config, denorm_factors, min_bounds, norm_type = \
        load_config_and_model(args.checkpoint, args.config, device)
    hybrid_model.with_residual = False
    print("  Residual disabled — using physics prior only for MPPI planning.\n")

    # --- Hydra env config ---
    try:
        from hydra import compose, initialize_config_dir
        config_dir = str(Path(__file__).parent / '../config')
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='default')
    except Exception as e:
        print(f"Warning: hydra config load failed ({e}), using minimal config")
        cfg = OmegaConf.create({'env': {}})

    # --- Collection loop ---
    print(f"\n{'='*100}")
    print("MPPI (PRIOR-ONLY) DATA GENERATION")
    print(f"  Trajectories : {args.num_trajectories}")
    print(f"  Steps/traj   : {args.num_steps}  ({args.num_steps / 100:.0f} s)")
    print(f"  MPPI samples : {args.mppi_samples}  |  Horizon: {args.mppi_horizon}")
    print(f"  Roll: [{ROLL_MIN}, {ROLL_MAX}]°  |  Pitch: [{PITCH_MIN}, {PITCH_MAX}]°")
    print(f"  Output       : {args.output}")
    print(f"{'='*100}\n")

    trajectory_results = []

    for traj_num in tqdm(range(1, args.num_trajectories + 1), desc="Trajectories", unit="traj"):
        print(f"--- Trajectory {traj_num}/{args.num_trajectories} ---")

        cfg.env.jsbsim = OmegaConf.load(
            str(Path(__file__).parent / '../config/env/jsbsim/noatmo.yaml')
        )
        env = initialize_environment(cfg)
        if env is None:
            print(f"  Failed to initialise environment — skipping trajectory {traj_num}")
            continue

        try:
            traj_data = run_single_trajectory(
                env,
                traj_num,
                hybrid_model,
                model_config,
                denorm_factors,
                min_bounds,
                norm_type,
                device,
                mppi_samples=args.mppi_samples,
                mppi_horizon=args.mppi_horizon,
                mppi_temperature=args.mppi_temperature,
                mppi_noise_std=args.mppi_noise_std,
                num_steps=args.num_steps,
            )
            trajectory_results.append(traj_data)
        except Exception as e:
            print(f"  Error in trajectory {traj_num}: {e}")
            import traceback
            traceback.print_exc()

    if not trajectory_results:
        print("No trajectories collected. Exiting.")
        return

    print_summary(trajectory_results)
    save_to_csv(trajectory_results, args.output)
    print("\n✓ MPPI prior-only data generation complete!")


if __name__ == '__main__':
    main()
