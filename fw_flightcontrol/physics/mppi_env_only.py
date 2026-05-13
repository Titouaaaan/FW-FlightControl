#!/usr/bin/env python3
"""
MPPI with JSBSim as the World Model (Ground-Truth Upper Bound)

Uses the actual JSBSim simulator — not the learned hybrid model — to roll out
candidate action sequences during MPPI planning.  This is the theoretical upper
bound on MPPI performance: if this works but model-based MPPI does not, the
learned dynamics model is the bottleneck, not the MPPI algorithm itself.

For each MPPI step the rollout env is sequentially reset to the current main-env
state (via IC save/restore) and stepped through each candidate action sequence.
This is slow (~1-2 s per control step) but is intended as a diagnostic tool.

Usage:
    cd fw_flightcontrol/physics/
    python mppi_env_only.py --target-roll 45 --target-pitch 25
    python mppi_env_only.py --target-roll 60 --target-pitch 30 --mppi-samples 30
"""

import numpy as np
import sys
import argparse
import time
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fw_jsbgym  # noqa: F401 — registers gym environments
import gymnasium as gym
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_flightcontrol.physics.utils import (
    get_env_state,
    set_env_state,
    plot_tracking_performance,
    plot_action_history,
    compute_convergence_stats,
)
from fw_flightcontrol.physics.mppi_test import MPPIController


# ============================================================================
# CONFIGURATION
# ============================================================================
MPPI_SAMPLES     = 50    # fewer samples — env rollouts are sequential and slow
MPPI_HORIZON     = 10
MPPI_TEMPERATURE = 1.0
MPPI_NOISE_STD   = 0.8
MAX_STEPS        = 1500
DT               = 0.01


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================
def make_env() -> gym.Env:
    try:
        from hydra import compose, initialize_config_dir
        config_dir = str(Path(__file__).parent / '../config')
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='default')
    except Exception:
        cfg = OmegaConf.create({'env': {}})

    cfg.env.jsbsim = OmegaConf.load(
        str(Path(__file__).parent / '../config/env/jsbsim/noatmo.yaml')
    )
    env = gym.make('ACBohnNoVaIErr-v0', cfg_env=cfg.env, render_mode='none')
    env.unwrapped.init()
    return env


# ============================================================================
# ENV-BASED TRAJECTORY ROLLOUT
# ============================================================================
def rollout_env_trajectories(
    rollout_env: gym.Env,
    saved_state: dict,
    actions: np.ndarray,        # (N, H, 3)
    target_roll_rad: float,
    target_pitch_rad: float,
) -> np.ndarray:                # (N,) costs
    """
    Roll out N candidate action sequences in the JSBSim environment.

    For each candidate: restore rollout_env to saved_state, apply the H-step
    action sequence, accumulate roll+pitch tracking cost.  The throttle is
    injected via the same monkey-patch used in mppi_test.py.
    """
    num_samples, horizon, _ = actions.shape
    costs = np.zeros(num_samples)

    # Throttle monkey-patch — same mechanism as mppi_test.run_mppi_control
    throttle_ref = [0.3]
    _original_apply = rollout_env.unwrapped.apply_action

    def _patched_apply(action):
        _original_apply(action)
        rollout_env.unwrapped.sim[prp.throttle_cmd] = float(
            np.clip(throttle_ref[0], 0.0, 1.0)
        )

    rollout_env.unwrapped.apply_action = _patched_apply

    for k in range(num_samples):
        set_env_state(rollout_env, saved_state)
        cost = 0.0

        for h in range(horizon):
            throttle_ref[0] = float(actions[k, h, 2])
            obs, _, terminated, truncated, _ = rollout_env.step(actions[k, h, :2])

            roll_err  = abs(obs[0] - target_roll_rad)
            pitch_err = abs(obs[1] - target_pitch_rad)
            cost += roll_err + pitch_err

            if terminated or truncated:
                cost += 100.0  # large penalty for crash
                break

        costs[k] = cost

    rollout_env.unwrapped.apply_action = _original_apply
    return costs


# ============================================================================
# MPPI OPTIMIZE (env-based, replaces MPPIController.optimize)
# ============================================================================
def mppi_optimize_env(
    controller: MPPIController,
    saved_state: dict,
    rollout_env: gym.Env,
    target_roll: float,
    target_pitch: float,
) -> np.ndarray:
    """One MPPI optimisation step using JSBSim rollouts."""
    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)

    # Sample action sequences (same logic as MPPIController.optimize)
    n_random = controller.num_samples // 4
    n_noise  = controller.num_samples - n_random

    noise          = np.random.normal(0, controller.noise_std,
                                      (n_noise, controller.horizon, controller.action_dim))
    exploit_actions = np.clip(controller.mean_actions[None] + noise, -1.0, 1.0)
    explore_actions = np.random.uniform(-1.0, 1.0,
                                        (n_random, controller.horizon, controller.action_dim))
    explore_actions[:, :, 2] = np.random.uniform(0.0, 1.0, (n_random, controller.horizon))
    sampled_actions = np.concatenate([exploit_actions, explore_actions], axis=0)
    sampled_actions[:, :, 2] = np.clip(sampled_actions[:, :, 2], 0.0, 1.0)

    # Evaluate all candidates in JSBSim
    costs = rollout_env_trajectories(
        rollout_env, saved_state, sampled_actions,
        target_roll_rad, target_pitch_rad,
    )

    # MPPI weights
    valid  = np.isfinite(costs)
    if not valid.any():
        controller._shift_mean()
        return np.zeros(controller.action_dim)

    nan_pen     = np.nanmax(costs[valid]) * 10.0
    costs_safe  = np.where(valid, costs, nan_pen)
    shifted     = costs_safe - costs_safe.min()
    weights     = np.exp(-shifted / controller.temperature)
    weights    /= weights.sum()

    controller.mean_actions = np.einsum('k,khd->hd', weights, sampled_actions)

    best = controller.mean_actions[0].copy()
    best[:2] = np.clip(best[:2], -1.0, 1.0)
    best[2]  = np.clip(best[2],  0.0,  1.0)
    controller._shift_mean()
    return best


# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================
def run_mppi_env_control(
    main_env: gym.Env,
    rollout_env: gym.Env,
    target_roll: float,
    target_pitch: float,
    max_steps: int,
    mppi_samples: int,
    mppi_horizon: int,
    mppi_temperature: float,
    mppi_noise_std: float,
) -> tuple:
    print(f"\n{'='*70}")
    print("MPPI CONTROL — JSBSim World Model")
    print(f"  Target: Roll={target_roll:+.1f}°  Pitch={target_pitch:+.1f}°")
    print(f"  MPPI: samples={mppi_samples}  horizon={mppi_horizon}  "
          f"temp={mppi_temperature}  noise={mppi_noise_std}")
    print(f"{'='*70}\n")

    # Throttle monkey-patch for main env
    throttle_ref = [0.3]
    _orig_apply = main_env.unwrapped.apply_action

    def _patched(action):
        _orig_apply(action)
        main_env.unwrapped.sim[prp.throttle_cmd] = float(
            np.clip(throttle_ref[0], 0.0, 1.0)
        )

    main_env.unwrapped.apply_action = _patched

    controller = MPPIController(
        horizon=mppi_horizon,
        action_dim=3,
        num_samples=mppi_samples,
        temperature=mppi_temperature,
        noise_std=mppi_noise_std,
    )
    controller.reset()

    main_env.unwrapped.init()
    obs, _ = main_env.reset()

    roll_hist    = []
    pitch_hist   = []
    actions_hist = []
    step_times   = []

    print(f"{'Step':>5} | {'Roll':>8} {'Err':>6} | {'Pitch':>8} {'Err':>6} | "
          f"{'Action [a,e,t]':>20} | {'Time(s)':>8}")
    print("-" * 80)

    for step in range(max_steps):
        t0 = time.time()

        saved_state = get_env_state(main_env)
        best_action = mppi_optimize_env(
            controller, saved_state, rollout_env,
            target_roll, target_pitch,
        )

        throttle_ref[0] = float(best_action[2])
        obs, _, terminated, truncated, _ = main_env.step(best_action[:2])
        elapsed = time.time() - t0

        roll_deg  = np.rad2deg(obs[0])
        pitch_deg = np.rad2deg(obs[1])
        roll_err  = abs(roll_deg  - target_roll)
        pitch_err = abs(pitch_deg - target_pitch)

        roll_hist.append(roll_deg)
        pitch_hist.append(pitch_deg)
        actions_hist.append(best_action.copy())
        step_times.append(elapsed)

        print(f"{step+1:>5} | {roll_deg:>8.2f} {roll_err:>6.2f} | {pitch_deg:>8.2f} {pitch_err:>6.2f} | "
              f"[{best_action[0]:>6.3f},{best_action[1]:>6.3f},{best_action[2]:>5.3f}] | {elapsed:>8.3f}")

        if terminated or truncated:
            print(f"\n>>> Terminated at step {step+1}")
            break

    main_env.unwrapped.apply_action = _orig_apply

    avg_t = np.mean(step_times)
    roll_conv  = compute_convergence_stats(roll_hist,  target_roll,  threshold_deg=1.0)
    pitch_conv = compute_convergence_stats(pitch_hist, target_pitch, threshold_deg=1.0)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"  Steps: {len(roll_hist)}  |  Avg step time: {avg_t:.3f}s")
    print(f"  Roll  — mean err: {np.mean(np.abs(np.array(roll_hist)-target_roll)):.2f}°  "
          f"converge: {roll_conv['convergence_step']}")
    print(f"  Pitch — mean err: {np.mean(np.abs(np.array(pitch_hist)-target_pitch)):.2f}°  "
          f"converge: {pitch_conv['convergence_step']}")
    print(f"{'='*70}\n")

    return roll_hist, pitch_hist, np.array(actions_hist)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='MPPI with JSBSim as world model (ground-truth upper bound)'
    )
    parser.add_argument('--target-roll',       type=float, default=30.0)
    parser.add_argument('--target-pitch',      type=float, default=15.0)
    parser.add_argument('--steps',             type=int,   default=MAX_STEPS)
    parser.add_argument('--mppi-samples',      type=int,   default=MPPI_SAMPLES)
    parser.add_argument('--mppi-horizon',      type=int,   default=MPPI_HORIZON)
    parser.add_argument('--mppi-temperature',  type=float, default=MPPI_TEMPERATURE)
    parser.add_argument('--mppi-noise-std',    type=float, default=MPPI_NOISE_STD)
    parser.add_argument('--plot-name',         type=str,   default=None)
    args = parser.parse_args()

    print("Initialising main env...")
    main_env = make_env()
    print("Initialising rollout env...")
    rollout_env = make_env()

    roll_hist, pitch_hist, actions = run_mppi_env_control(
        main_env   = main_env,
        rollout_env= rollout_env,
        target_roll      = args.target_roll,
        target_pitch     = args.target_pitch,
        max_steps        = args.steps,
        mppi_samples     = args.mppi_samples,
        mppi_horizon     = args.mppi_horizon,
        mppi_temperature = args.mppi_temperature,
        mppi_noise_std   = args.mppi_noise_std,
    )

    main_env.close()
    rollout_env.close()

    tag = args.plot_name or f"env_mppi_r{args.target_roll:.0f}_p{args.target_pitch:.0f}"
    plot_tracking_performance(
        target_roll=args.target_roll, target_pitch=args.target_pitch,
        filename=f"{tag}_tracking.png",
        mppi_roll=roll_hist, mppi_pitch=pitch_hist,
    )
    plot_action_history(filename=f"{tag}_actions.png", mppi_actions=actions)


if __name__ == '__main__':
    main()
