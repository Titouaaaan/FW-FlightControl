#!/usr/bin/env python3
"""
PID Performance on Hard Targets

Runs PID roll/pitch tracking on 10 challenging target combinations for 2000 steps
each, then produces a single multi-panel figure showing all tracking results.

Usage:
    cd fw_flightcontrol/physics/
    python test_pid_performance.py
"""

import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fw_jsbgym  # noqa: F401 — registers JSBSim gym environments
import gymnasium as gym
from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.utils import jsbsim_properties as prp


# ============================================================================
# CONFIGURATION
# ============================================================================
HARD_TARGETS = [
    ( 45.0,  25.0),
    (-45.0,  25.0),
    ( 55.0,  27.0),
    (-55.0,  27.0),
    ( 60.0,  30.0),
    (-60.0,  30.0),
    ( 60.0, -27.0),
    (-60.0, -27.0),
    ( 47.0,  30.0),
    (-47.0,  30.0),
]

NUM_STEPS  = 2000
DT         = 0.01
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'pid-performance'
OUTPUT_FILE = 'pid_hard_targets.png'


# ============================================================================
# ENVIRONMENT INIT
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
    return gym.make('ACBohnNoVaIErr-v0', cfg_env=cfg.env, render_mode='none')


# ============================================================================
# SINGLE EPISODE
# ============================================================================
def run_pid_episode(target_roll_deg: float, target_pitch_deg: float) -> dict:
    env = make_env()
    env.unwrapped.init()
    obs, _ = env.reset()

    target_roll_rad  = np.deg2rad(target_roll_deg)
    target_pitch_rad = np.deg2rad(target_pitch_deg)

    pid_aileron = PID(
        kp=1.5, ki=0.1, kd=0.1,
        dt=env.unwrapped.fdm_dt,
        trim=TrimPoint(), limit=1.0, is_throttle=False,
    )
    pid_elevator = PID(
        kp=-2.0, ki=-0.3, kd=-0.1,
        dt=env.unwrapped.fdm_dt,
        trim=TrimPoint(), limit=1.0, is_throttle=False,
    )
    pid_aileron.set_reference(target_roll_rad)
    pid_elevator.set_reference(target_pitch_rad)

    roll_hist  = []
    pitch_hist = []

    for _ in range(NUM_STEPS):
        roll, pitch   = obs[0], obs[1]
        p_radps, q_radps = obs[3], obs[4]

        aileron_cmd,  _, _ = pid_aileron.update(state=roll,  state_dot=p_radps, saturate=True, normalize=False)
        elevator_cmd, _, _ = pid_elevator.update(state=pitch, state_dot=q_radps, saturate=True, normalize=False)

        obs, _, terminated, truncated, _ = env.step(np.array([aileron_cmd, elevator_cmd]))
        roll_hist.append(np.rad2deg(obs[0]))
        pitch_hist.append(np.rad2deg(obs[1]))

        if terminated or truncated:
            break

    env.close()

    avg_roll_err  = float(np.mean(np.abs(np.array(roll_hist)  - target_roll_deg)))
    avg_pitch_err = float(np.mean(np.abs(np.array(pitch_hist) - target_pitch_deg)))
    return {
        'roll':           roll_hist,
        'pitch':          pitch_hist,
        'target_roll':    target_roll_deg,
        'target_pitch':   target_pitch_deg,
        'avg_roll_err':   avg_roll_err,
        'avg_pitch_err':  avg_pitch_err,
        'steps':          len(roll_hist),
    }


# ============================================================================
# MULTI-PANEL PLOT
# ============================================================================
def plot_all(results: list, output_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), sharex=False)

    time = np.arange(NUM_STEPS) * DT

    for i, res in enumerate(results):
        ax_roll  = axes[i, 0]
        ax_pitch = axes[i, 1]
        t = time[:res['steps']]

        # Roll
        ax_roll.plot(t, res['roll'], color='darkorange', linewidth=1.1)
        ax_roll.axhline(res['target_roll'], color='red', linestyle='--', linewidth=1.2, label='Target')
        ax_roll.set_ylabel('Roll [°]')
        ax_roll.set_title(
            f"Roll={res['target_roll']:+.0f}°  Pitch={res['target_pitch']:+.0f}°  "
            f"(err: roll={res['avg_roll_err']:.1f}°)",
            fontsize=9,
        )
        ax_roll.legend(loc='upper right', fontsize=8)
        ax_roll.grid(True, alpha=0.3)

        # Pitch
        ax_pitch.plot(t, res['pitch'], color='steelblue', linewidth=1.1)
        ax_pitch.axhline(res['target_pitch'], color='red', linestyle='--', linewidth=1.2, label='Target')
        ax_pitch.set_ylabel('Pitch [°]')
        ax_pitch.set_title(
            f"Roll={res['target_roll']:+.0f}°  Pitch={res['target_pitch']:+.0f}°  "
            f"(err: pitch={res['avg_pitch_err']:.1f}°)",
            fontsize=9,
        )
        ax_pitch.legend(loc='upper right', fontsize=8)
        ax_pitch.grid(True, alpha=0.3)

        if i == n - 1:
            ax_roll.set_xlabel('Time [s]')
            ax_pitch.set_xlabel('Time [s]')

    fig.suptitle('PID Tracking Performance — Hard Targets', fontsize=13, fontweight='bold', y=1.002)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"Running PID on {len(HARD_TARGETS)} hard targets ({NUM_STEPS} steps each)\n")
    print(f"{'#':>3}  {'Roll':>6}  {'Pitch':>6}  {'Roll Err':>9}  {'Pitch Err':>10}  {'Steps':>6}")
    print("-" * 55)

    results = []
    for i, (roll_target, pitch_target) in enumerate(HARD_TARGETS, 1):
        res = run_pid_episode(roll_target, pitch_target)
        results.append(res)
        print(
            f"{i:>3}  {roll_target:>+6.1f}  {pitch_target:>+6.1f}  "
            f"{res['avg_roll_err']:>9.2f}°  {res['avg_pitch_err']:>10.2f}°  {res['steps']:>6}"
        )

    plot_all(results, OUTPUT_DIR / OUTPUT_FILE)
    print("\nDone.")


if __name__ == '__main__':
    main()
