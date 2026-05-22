#!/usr/bin/env python3
"""
Full MPPI Analysis — side-by-side comparison of all world models.

Runs the same episode with identical MPPI settings for:
  - PID baseline
  - Physics prior only (residual zeroed out)
  - Each .pt model found in physics/models/
  - JSBSim environment as world model (optional, slow)

For each run, saves a 4-panel figure to data/full_analysis/:
  top-left:     roll control
  top-right:    pitch control
  bottom-left:  commands (aileron / elevator / throttle)
  bottom-right: angular velocities (p / q / r)

Also saves a metrics CSV with convergence, mean error, and timing stats.

Usage:
    cd fw_flightcontrol/physics/tests/
    python full_mppi_analysis.py
    python full_mppi_analysis.py --target-roll 55 --target-pitch 28 --steps 200
    python full_mppi_analysis.py --skip-env-model
"""

import numpy as np
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from omegaconf import OmegaConf

import fw_jsbgym  # noqa: F401 — registers gym environments
import gymnasium as gym
import torch

from fw_jsbgym.utils import jsbsim_properties as prp
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_flightcontrol.agents.pid import PID
from fw_flightcontrol.physics.utils import (
    get_env_state,
    set_env_state,
    get_norm_type,
    clean_state_dict_for_compilation,
    suppress_output,
    _throttle_patch,
    _safe_label,
    plot_model_result,
    save_metrics,
)
from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.tests.mppi_test import MPPIController, rollout_trajectories

# ============================================================================
# PATHS
# ============================================================================
_TESTS_DIR    = Path(__file__).parent
_PHYSICS_DIR  = _TESTS_DIR.parent
_FC_DIR       = _PHYSICS_DIR.parent
MODELS_DIR    = _PHYSICS_DIR / 'models'
CONFIG_DIR    = str(_FC_DIR / 'config')
NOATMO_YAML   = str(_FC_DIR / 'config' / 'env' / 'jsbsim' / 'noatmo.yaml')
TRAINING_YAML = str(_PHYSICS_DIR / 'training_params.yaml')
SAVE_DIR      = _FC_DIR / 'data' / 'full_analysis_nominal'

DT        = 0.01
STEPS_20S = 2000   # 20 s at 0.01 s/step


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def make_env() -> gym.Env:
    try:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name='default')
    except Exception:
        cfg = OmegaConf.create({'env': {}})
    cfg.env.jsbsim = OmegaConf.load(NOATMO_YAML)
    env = gym.make('ACBohnNoVaIErr-v0', cfg_env=cfg.env, render_mode='none')
    env.unwrapped.init()
    return env


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: str, device: torch.device):
    import yaml
    with open(TRAINING_YAML) as f:
        config = yaml.safe_load(f)

    physics_prior = PhysicsPrior()
    with suppress_output():
        raw = torch.load(model_path, map_location=device)

    denorm_factors = min_bounds = None
    if isinstance(raw, dict) and 'residual_state' in raw:
        residual_state = clean_state_dict_for_compilation(raw['residual_state'])
        if 'norm_scale' in raw and 'norm_offset' in raw:
            denorm_factors = torch.tensor(raw['norm_scale'], dtype=torch.float32, device=device)
            min_bounds     = torch.tensor(raw['norm_offset'], dtype=torch.float32, device=device)
    else:
        residual_state = clean_state_dict_for_compilation(raw)

    def _infer_hidden(sd):
        dims, i = [], 0
        while f'network.{i}.weight' in sd:
            if f'network.{i+2}.weight' in sd:
                dims.append(sd[f'network.{i}.weight'].shape[0])
            i += 2
        return dims

    net = config['network']
    hidden = _infer_hidden(residual_state) or net['hidden_dims']

    residual = PhysicsAugmented(
        state_dim=net['state_dim'], action_dim=net['action_dim'],
        hidden_dims=hidden, activation=net['activation'],
        use_batch_norm=net['use_batch_norm'],
    )
    residual.load_state_dict(residual_state)

    method = config.get('integration', {}).get('method', 'rk4')
    hybrid = HybridDynamicsModel(
        physics_prior=physics_prior, residual_network=residual,
        with_prior=True, with_residual=True, integration_method=method,
    ).to(device).eval()

    norm_type = get_norm_type(config)
    if norm_type is not None and denorm_factors is None:
        p = config.get('normalization_params', {})
        if p:
            denorm_factors = torch.tensor(p['norm_scale'], dtype=torch.float32, device=device)
            min_bounds     = torch.tensor(p['norm_offset'], dtype=torch.float32, device=device)
        else:
            norm_type = None

    return hybrid, config, denorm_factors, min_bounds, norm_type


# ============================================================================
# PID RUN
# ============================================================================

def run_pid(target_roll: float, target_pitch: float, max_steps: int, seed: int) -> dict:
    np.random.seed(seed)
    with suppress_output():
        env = make_env()
        obs, _ = env.reset()

    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)

    pid_ail = PID(kp=1.5,  ki=0.1,  kd=0.1,  dt=env.unwrapped.fdm_dt,
                  trim=TrimPoint(), limit=1.0, is_throttle=False)
    pid_ele = PID(kp=-2.0, ki=-0.3, kd=-0.1, dt=env.unwrapped.fdm_dt,
                  trim=TrimPoint(), limit=1.0, is_throttle=False)
    pid_ail.set_reference(target_roll_rad)
    pid_ele.set_reference(target_pitch_rad)

    roll_hist, pitch_hist, p_hist, q_hist, r_hist = [], [], [], [], []
    actions_hist, step_times = [], []
    terminated = False

    with tqdm(total=max_steps, desc='PID', unit='step', leave=True) as pbar:
        for _ in range(max_steps):
            t0 = time.time()
            roll, pitch      = obs[0], obs[1]
            p_radps, q_radps = obs[3], obs[4]

            ail, _, _ = pid_ail.update(state=roll,  state_dot=p_radps, saturate=True, normalize=False)
            ele, _, _ = pid_ele.update(state=pitch, state_dot=q_radps, saturate=True, normalize=False)
            throttle  = float(env.unwrapped.sim[prp.throttle_cmd])

            obs, _, term, trunc, _ = env.step(np.array([ail, ele]))

            roll_hist.append(np.rad2deg(obs[0]))
            pitch_hist.append(np.rad2deg(obs[1]))
            p_hist.append(obs[3])
            q_hist.append(obs[4])
            r_hist.append(obs[5])
            actions_hist.append([ail, ele, throttle])
            step_times.append(time.time() - t0)

            pbar.set_postfix(roll=f'{roll_hist[-1]:+.1f}°', pitch=f'{pitch_hist[-1]:+.1f}°')
            pbar.update(1)

            if term or trunc:
                terminated = True
                break

    env.close()
    return dict(label='PID', roll=roll_hist, pitch=pitch_hist,
                p=p_hist, q=q_hist, r=r_hist,
                actions=np.array(actions_hist), step_times=step_times,
                steps=len(roll_hist), terminated=terminated)


# ============================================================================
# ENV-BASED MPPI RUN
# ============================================================================

def _rollout_env(rollout_env, saved_state, actions, target_roll_rad, target_pitch_rad,
                 target_va_kmh, va_weight):
    num_samples, horizon, _ = actions.shape
    costs = np.zeros(num_samples)
    throttle_ref, restore = _throttle_patch(rollout_env)

    for k in range(num_samples):
        with suppress_output():
            rollout_env.reset()
        set_env_state(rollout_env, saved_state)
        cost = 0.0
        for h in range(horizon):
            throttle_ref[0] = float(actions[k, h, 2])
            obs, _, term, trunc, _ = rollout_env.step(actions[k, h, :2])
            cost += (abs(obs[0] - target_roll_rad)
                     + abs(obs[1] - target_pitch_rad)
                     + va_weight * abs(obs[2] - target_va_kmh))
            if term or trunc:
                cost += 100.0
                break
        costs[k] = cost

    restore()
    return costs


def run_mppi_env(target_roll: float, target_pitch: float, max_steps: int,
                 mppi_cfg: dict, seed: int) -> dict:
    with suppress_output():
        main_env    = make_env()
        rollout_env = make_env()

    throttle_ref, restore = _throttle_patch(main_env)
    controller = MPPIController(
        horizon=mppi_cfg['horizon'], action_dim=3,
        num_samples=mppi_cfg['samples'], temperature=mppi_cfg['temperature'],
        noise_std=mppi_cfg['noise_std'],
    )
    np.random.seed(seed)
    controller.reset()
    with suppress_output():
        obs, _ = main_env.reset()

    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)
    va_weight        = mppi_cfg.get('va_weight', 0.1)

    roll_hist, pitch_hist, p_hist, q_hist, r_hist = [], [], [], [], []
    actions_hist, step_times = [], []
    terminated = False

    with tqdm(total=max_steps, desc='MPPI env', unit='step', leave=True) as pbar:
        for _ in range(max_steps):
            t0 = time.time()
            saved     = get_env_state(main_env)
            target_va = saved['airspeed_kts'] * 1.852

            sampled = controller.sample_actions()
            costs   = _rollout_env(rollout_env, saved, sampled,
                                   target_roll_rad, target_pitch_rad, target_va, va_weight)

            valid = np.isfinite(costs)
            if not valid.any():
                controller._shift_mean()
                best = np.zeros(3)
            else:
                nan_pen = np.nanmax(costs[valid]) * 10.0
                cs = np.where(valid, costs, nan_pen)
                w  = np.exp(-(cs - cs.min()) / controller.temperature)
                w /= w.sum()
                controller.mean_actions = np.einsum('k,khd->hd', w, sampled)
                best = controller.mean_actions[0].copy()
                best[:2] = np.clip(best[:2], -1.0, 1.0)
                best[2]  = np.clip(best[2],  0.0,  1.0)
                controller._shift_mean()

            throttle_ref[0] = float(best[2])
            obs, _, term, trunc, _ = main_env.step(best[:2])

            roll_hist.append(np.rad2deg(obs[0]))
            pitch_hist.append(np.rad2deg(obs[1]))
            p_hist.append(obs[3])
            q_hist.append(obs[4])
            r_hist.append(obs[5])
            actions_hist.append(best.copy())
            step_times.append(time.time() - t0)

            pbar.set_postfix(roll=f'{roll_hist[-1]:+.1f}°', pitch=f'{pitch_hist[-1]:+.1f}°',
                             t=f'{step_times[-1]:.1f}s')
            pbar.update(1)

            if term or trunc:
                terminated = True
                break

    restore()
    main_env.close()
    rollout_env.close()
    return dict(label='JSBSim_env', roll=roll_hist, pitch=pitch_hist,
                p=p_hist, q=q_hist, r=r_hist,
                actions=np.array(actions_hist), step_times=step_times,
                steps=len(roll_hist), terminated=terminated)


# ============================================================================
# MODEL-BASED MPPI RUN
# ============================================================================

def run_mppi_model(label: str, model_path: Optional[str], target_roll: float,
                   target_pitch: float, max_steps: int, mppi_cfg: dict,
                   seed: int, device: torch.device,
                   residual_clamp: Optional[float] = None) -> dict:
    if model_path is not None:
        hybrid, config, denorm_factors, min_bounds, norm_type = load_model(model_path, device)
    else:
        import yaml
        with open(TRAINING_YAML) as f:
            config = yaml.safe_load(f)
        physics_prior = PhysicsPrior()
        net = config['network']
        residual = PhysicsAugmented(
            state_dim=net['state_dim'], action_dim=net['action_dim'],
            hidden_dims=net['hidden_dims'], activation=net['activation'],
            use_batch_norm=net['use_batch_norm'],
        )
        method = config.get('integration', {}).get('method', 'rk4')
        hybrid = HybridDynamicsModel(
            physics_prior=physics_prior, residual_network=residual,
            with_prior=True, with_residual=True, integration_method=method,
        ).to(device).eval()
        denorm_factors = min_bounds = norm_type = None
        residual_clamp = 0.0

    with suppress_output():
        env = make_env()
    throttle_ref, restore = _throttle_patch(env)
    controller = MPPIController(
        horizon=mppi_cfg['horizon'], action_dim=3,
        num_samples=mppi_cfg['samples'], temperature=mppi_cfg['temperature'],
        noise_std=mppi_cfg['noise_std'],
    )
    np.random.seed(seed)
    controller.reset()
    with suppress_output():
        obs, _ = env.reset()

    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)

    roll_hist, pitch_hist, p_hist, q_hist, r_hist = [], [], [], [], []
    actions_hist, step_times = [], []
    terminated = False

    with tqdm(total=max_steps, desc=label, unit='step', leave=True) as pbar:
        for _ in range(max_steps):
            t0      = time.time()
            sampled = controller.sample_actions()

            trajectories = rollout_trajectories(
                obs, sampled, hybrid, config,
                denorm_factors, min_bounds, norm_type, device,
                residual_clamp=residual_clamp,
            )

            roll_errs  = np.abs(trajectories[:, :, 0] - target_roll_rad)
            pitch_errs = np.abs(trajectories[:, :, 1] - target_pitch_rad)
            costs      = np.sum(roll_errs + pitch_errs, axis=1)

            valid = np.isfinite(costs)
            if not valid.any():
                controller._shift_mean()
                best = np.zeros(3)
            else:
                nan_pen = np.nanmax(costs[valid]) * 10.0
                cs = np.where(valid, costs, nan_pen)
                w  = np.exp(-(cs - cs.min()) / controller.temperature)
                w /= w.sum()
                controller.mean_actions = np.einsum('k,khd->hd', w, sampled)
                best = controller.mean_actions[0].copy()
                best[:2] = np.clip(best[:2], -1.0, 1.0)
                best[2]  = np.clip(best[2],  0.0,  1.0)
                controller._shift_mean()

            throttle_ref[0] = float(best[2])
            obs, _, term, trunc, _ = env.step(best[:2])

            roll_hist.append(np.rad2deg(obs[0]))
            pitch_hist.append(np.rad2deg(obs[1]))
            p_hist.append(obs[3])
            q_hist.append(obs[4])
            r_hist.append(obs[5])
            actions_hist.append(best.copy())
            step_times.append(time.time() - t0)

            pbar.set_postfix(roll=f'{roll_hist[-1]:+.1f}°', pitch=f'{pitch_hist[-1]:+.1f}°')
            pbar.update(1)

            if term or trunc:
                terminated = True
                break

    restore()
    env.close()
    return dict(label=label, roll=roll_hist, pitch=pitch_hist,
                p=p_hist, q=q_hist, r=r_hist,
                actions=np.array(actions_hist), step_times=step_times,
                steps=len(roll_hist), terminated=terminated)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full MPPI analysis — compare all world models side by side'
    )
    parser.add_argument('--target-roll',      type=float, default=20.0)
    parser.add_argument('--target-pitch',     type=float, default=10.0)
    parser.add_argument('--steps',            type=int,   default=STEPS_20S,
                        help='Steps per run (default 2000 = 20 s)')
    parser.add_argument('--mppi-samples',     type=int,   default=1000)
    parser.add_argument('--mppi-horizon',     type=int,   default=40)
    parser.add_argument('--mppi-temperature', type=float, default=0.5)
    parser.add_argument('--mppi-noise-std',   type=float, default=0.5)
    parser.add_argument('--mppi-va-weight',   type=float, default=0.1)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--device',           type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-env-model',   action='store_true',
                        help='Skip the slow JSBSim-rollout MPPI run')
    args = parser.parse_args()

    device   = torch.device(args.device)
    mppi_cfg = dict(
        samples=args.mppi_samples, horizon=args.mppi_horizon,
        temperature=args.mppi_temperature, noise_std=args.mppi_noise_std,
        va_weight=args.mppi_va_weight,
    )
    tag = f"r{args.target_roll:.0f}_p{args.target_pitch:.0f}"

    print(f"\nFull MPPI Analysis")
    print(f"  Target : Roll={args.target_roll:+.1f}°  Pitch={args.target_pitch:+.1f}°")
    print(f"  Steps  : {args.steps} ({args.steps * DT:.0f} s)")
    print(f"  MPPI   : samples={args.mppi_samples}  horizon={args.mppi_horizon}  "
          f"temp={args.mppi_temperature}  noise={args.mppi_noise_std}")
    print(f"  Seed   : {args.seed}   Device: {device}")

    models = sorted(MODELS_DIR.glob('*.pt'))
    print(f"  Models : {', '.join(m.stem for m in models) or 'none found'}\n")

    results: List[dict] = []

    results.append(run_pid(args.target_roll, args.target_pitch, args.steps, args.seed))

    results.append(run_mppi_model(
        label='Prior_only', model_path=None,
        target_roll=args.target_roll, target_pitch=args.target_pitch,
        max_steps=args.steps, mppi_cfg=mppi_cfg, seed=args.seed, device=device,
    ))

    for model_path in models:
        results.append(run_mppi_model(
            label=model_path.stem, model_path=str(model_path),
            target_roll=args.target_roll, target_pitch=args.target_pitch,
            max_steps=args.steps, mppi_cfg=mppi_cfg, seed=args.seed, device=device,
        ))

    if not args.skip_env_model:
        results.append(run_mppi_env(
            args.target_roll, args.target_pitch, args.steps, mppi_cfg, args.seed
        ))

    print(f"\nSaving plots to {SAVE_DIR}/")
    for res in results:
        fname = f"{_safe_label(res['label'])}_{tag}_plots.png"
        plot_model_result(res, args.target_roll, args.target_pitch, SAVE_DIR / fname)

    save_metrics(results, args.target_roll, args.target_pitch,
                 SAVE_DIR / f"metrics_{tag}.csv")

# example to run:
# python full_mppi_analysis.py --target-roll 55 --target-pitch 28 --mppi-samples 500 --steps 2000
if __name__ == '__main__':
    main()
