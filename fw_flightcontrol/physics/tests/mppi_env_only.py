#!/usr/bin/env python3
"""
MPPI with JSBSim as the World Model (Ground-Truth Upper Bound)

Standalone version of the env-model run from full_mppi_analysis.py.
All code is taken verbatim from that script — same paths, constants,
make_env, _rollout_env, and run_mppi_env — so results are directly comparable.

Usage:
    cd fw_flightcontrol/physics/tests/
    python mppi_env_only.py
    python mppi_env_only.py --target-roll 55 --target-pitch 28 --steps 200
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
from fw_flightcontrol.physics.tests.mppi_test import MPPIController

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
SAVE_DIR      = _FC_DIR / 'data' / 'fixing_sim_env'

DT           = 0.01
STEPS_20S    = 2000   # 20 s at 0.01 s/step
TARGET_VA_KPH = 60.0  # fixed cruise-speed target [km/h]


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def make_env(render_mode: str = 'none', telemetry_file: str = '') -> gym.Env:
    try:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name='default')
    except Exception:
        cfg = OmegaConf.create({'env': {}})
    cfg.env.jsbsim = OmegaConf.load(NOATMO_YAML)
    import os
    if render_mode != 'none' and not telemetry_file:
        os.makedirs('telemetry', exist_ok=True)
        telemetry_file = 'telemetry/telemetry.csv'
    env = gym.make('ACBohnNoVaIErr-v0', cfg_env=cfg.env,
                   render_mode=render_mode, telemetry_file=telemetry_file)
    env.unwrapped.init()
    return env


# ============================================================================
# STATE DIAGNOSTICS
# ============================================================================

# How many MPPI steps (from step 0) to run full state diagnostics on.
_DIAG_STEPS = 2


def _dump_full_state(label: str, env) -> None:
    """Print all JSBSim physical + extended + Python-wrapper state for one env."""
    sim = env.unwrapped.sim
    uw  = env.unwrapped
    print(f"\n  [{label}]")

    print(f"    ── JSBSim physical state (saved/restored by get/set_env_state) ──")
    print(f"    roll         : {sim[prp.roll_rad]:+.5f} rad  ({np.rad2deg(sim[prp.roll_rad]):+.2f}°)")
    print(f"    pitch        : {sim[prp.pitch_rad]:+.5f} rad  ({np.rad2deg(sim[prp.pitch_rad]):+.2f}°)")
    print(f"    heading      : {sim[prp.heading_rad]:+.5f} rad")
    print(f"    airspeed     : {sim[prp.airspeed_kts]:.5f} kts  ({sim[prp.airspeed_kph]:.5f} kph)")
    print(f"    p / q / r    : {sim[prp.p_radps]:+.5f} / {sim[prp.q_radps]:+.5f} / {sim[prp.r_radps]:+.5f} rad/s")
    print(f"    altitude     : {sim[prp.altitude_sl_ft]:.3f} ft")
    print(f"    alpha / beta : {sim[prp.alpha_rad]:+.5f} / {sim[prp.beta_rad]:+.5f} rad")
    print(f"    aileron_cmd  : {sim[prp.aileron_cmd]:+.5f}")
    print(f"    elevator_cmd : {sim[prp.elevator_cmd]:+.5f}")
    print(f"    throttle_cmd : {sim[prp.throttle_cmd]:+.5f}")

    print(f"    ── Extended JSBSim state (NOT saved/restored) ──────────────────")
    print(f"    target_roll  : {sim[prp.target_roll_rad]:+.5f} rad  ({np.rad2deg(sim[prp.target_roll_rad]):+.2f}°)")
    print(f"    target_pitch : {sim[prp.target_pitch_rad]:+.5f} rad  ({np.rad2deg(sim[prp.target_pitch_rad]):+.2f}°)")
    print(f"    roll_err     : {sim[prp.roll_err]:+.5f} rad")
    print(f"    pitch_err    : {sim[prp.pitch_err]:+.5f} rad")
    print(f"    roll_integ   : {sim[prp.roll_integ_err]:+.7f} rad")
    print(f"    pitch_integ  : {sim[prp.pitch_integ_err]:+.7f} rad")
    if hasattr(uw, 'current_step'):
        try:
            print(f"    current_step : {sim[uw.current_step]:.0f}")
        except Exception:
            pass

    print(f"    ── Python wrapper state (NOT saved/restored) ───────────────────")
    if hasattr(uw, 'pid_airspeed'):
        print(f"    pid_Va.integ : {uw.pid_airspeed.integral:+.7f}")
        print(f"    pid_Va.ref   : {uw.pid_airspeed.ref:+.5f}")
    if hasattr(uw, 'action_hist') and len(uw.action_hist) > 0:
        last = np.array(uw.action_hist)[-1]
        last_str = '  '.join(f'{v:+.4f}' for v in last)
        print(f"    action_hist  : len={len(uw.action_hist)}  last=[{last_str}]")
    else:
        print(f"    action_hist  : (empty)")
    if hasattr(uw, 'observation_deque'):
        print(f"    obs_deque    : len={len(uw.observation_deque)}")


def _print_state_diff(main_env, rollout_env) -> None:
    """Compare main_env vs rollout_env state field by field and flag mismatches."""
    sim_m = main_env.unwrapped.sim
    sim_r = rollout_env.unwrapped.sim
    uw_m  = main_env.unwrapped
    uw_r  = rollout_env.unwrapped
    tol   = 1e-4

    def _row(name, mv, rv):
        diff  = abs(mv - rv)
        flag  = "  ← MISMATCH" if diff > tol else ""
        print(f"    {name:<18}: main={mv:+.5f}  rollout={rv:+.5f}  |diff|={diff:.2e}{flag}")

    print(f"\n  [DIFF main_env vs rollout_env after set_env_state]")
    print(f"    ── physical (expect zero diff) ─────────────────────────────────")
    _row("roll_rad",      sim_m[prp.roll_rad],      sim_r[prp.roll_rad])
    _row("pitch_rad",     sim_m[prp.pitch_rad],     sim_r[prp.pitch_rad])
    _row("heading_rad",   sim_m[prp.heading_rad],   sim_r[prp.heading_rad])
    _row("airspeed_kts",  sim_m[prp.airspeed_kts],  sim_r[prp.airspeed_kts])
    _row("p_radps",       sim_m[prp.p_radps],       sim_r[prp.p_radps])
    _row("q_radps",       sim_m[prp.q_radps],       sim_r[prp.q_radps])
    _row("r_radps",       sim_m[prp.r_radps],       sim_r[prp.r_radps])
    _row("altitude_ft",   sim_m[prp.altitude_sl_ft], sim_r[prp.altitude_sl_ft])
    _row("alpha_rad",     sim_m[prp.alpha_rad],     sim_r[prp.alpha_rad])
    _row("beta_rad",      sim_m[prp.beta_rad],      sim_r[prp.beta_rad])
    _row("aileron_cmd",   sim_m[prp.aileron_cmd],   sim_r[prp.aileron_cmd])
    _row("elevator_cmd",  sim_m[prp.elevator_cmd],  sim_r[prp.elevator_cmd])
    _row("throttle_cmd",  sim_m[prp.throttle_cmd],  sim_r[prp.throttle_cmd])
    print(f"    ── extended JSBSim (expect diff — not restored) ────────────────")
    _row("target_roll",   sim_m[prp.target_roll_rad],  sim_r[prp.target_roll_rad])
    _row("target_pitch",  sim_m[prp.target_pitch_rad], sim_r[prp.target_pitch_rad])
    _row("roll_err",      sim_m[prp.roll_err],      sim_r[prp.roll_err])
    _row("pitch_err",     sim_m[prp.pitch_err],     sim_r[prp.pitch_err])
    _row("roll_integ",    sim_m[prp.roll_integ_err],  sim_r[prp.roll_integ_err])
    _row("pitch_integ",   sim_m[prp.pitch_integ_err], sim_r[prp.pitch_integ_err])
    print(f"    ── Python wrapper (expect diff — not restored) ─────────────────")
    if hasattr(uw_m, 'pid_airspeed') and hasattr(uw_r, 'pid_airspeed'):
        _row("pid_Va.integ",  uw_m.pid_airspeed.integral, uw_r.pid_airspeed.integral)
    if hasattr(uw_m, 'action_hist') and hasattr(uw_r, 'action_hist'):
        last_m = np.array(uw_m.action_hist)[-1] if len(uw_m.action_hist) else np.zeros(2)
        last_r = np.array(uw_r.action_hist)[-1] if len(uw_r.action_hist) else np.zeros(2)
        for i, name in enumerate([f"action_hist[-1][{i}]" for i in range(len(last_m))]):
            _row(name, last_m[i], last_r[i])


# ============================================================================
# ENV-BASED MPPI RUN
# ============================================================================

def _rollout_env(rollout_env, saved_state, actions, target_roll_rad, target_pitch_rad,
                 target_va_kmh, va_weight, _diag_main_env=None, _diag_mppi_step=-1):
    num_samples, horizon, _ = actions.shape
    costs = np.zeros(num_samples)
    throttle_ref, restore = _throttle_patch(rollout_env)

    for k in range(num_samples):
        #with suppress_output():
        rollout_env.reset()

        if _diag_main_env is not None and k == 0:
            print(f"\n{'='*70}")
            print(f"STATE RESTORE DIAGNOSTIC — MPPI step {_diag_mppi_step}, sample {k}")
            print(f"{'='*70}")
            _dump_full_state("MAIN ENV (state at save time)", _diag_main_env)
            _dump_full_state("ROLLOUT ENV — after reset(), before set_env_state()", rollout_env)

        set_env_state(rollout_env, saved_state)

        if _diag_main_env is not None and k == 0:
            _dump_full_state("ROLLOUT ENV — after set_env_state()", rollout_env)
            _print_state_diff(_diag_main_env, rollout_env)
            print(f"\n{'='*70}\n")

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
                 mppi_cfg: dict, seed: int, render_mode: str = 'none') -> dict:
    with suppress_output():
        main_env    = make_env(render_mode=render_mode, telemetry_file='telemetry/mppi_env.csv')
        rollout_env = make_env()

    throttle_ref, restore = _throttle_patch(main_env)
    controller = MPPIController(
        horizon=mppi_cfg['horizon'], action_dim=3,
        num_samples=mppi_cfg['samples'], temperature=mppi_cfg['temperature'],
        noise_std=mppi_cfg['noise_std'],
    )
    np.random.seed(seed)
    controller.reset()
    obs, _ = main_env.reset(options={"fgear_target_roll": target_roll, "fgear_target_pitch": target_pitch})

    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)
    va_weight        = mppi_cfg.get('va_weight', 0.1)
    main_env.unwrapped.set_target_state(np.array([target_roll_rad, target_pitch_rad]))

    roll_hist, pitch_hist, va_hist, p_hist, q_hist, r_hist = [], [], [], [], [], []
    actions_hist, step_times = [], []
    terminated = False

    n_iters = mppi_cfg.get('iters') or 1

    with tqdm(total=max_steps, desc='MPPI env', unit='step', leave=True) as pbar:
        for step_number in range(max_steps):
            t0    = time.time()
            saved = get_env_state(main_env)
            best  = np.zeros(3)

            _diag = main_env if step_number < _DIAG_STEPS else None

            for _ in range(n_iters):
                sampled = controller.sample_actions()
                costs   = _rollout_env(rollout_env, saved, sampled,
                                       target_roll_rad, target_pitch_rad, TARGET_VA_KPH, va_weight,
                                       _diag_main_env=_diag, _diag_mppi_step=step_number)

                valid = np.isfinite(costs)
                if not valid.any():
                    continue
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
            va_hist.append(obs[2])
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
    return dict(label='JSBSim_env', roll=roll_hist, pitch=pitch_hist, va=va_hist,
                p=p_hist, q=q_hist, r=r_hist,
                actions=np.array(actions_hist), step_times=step_times,
                steps=len(roll_hist), terminated=terminated)


# ============================================================================
# DEEP STATE CLONE
# ============================================================================

def deep_clone_state(src_env, dst_env) -> dict:
    """
    Fully clone src_env into dst_env.

    Three layers applied in order:
      1. set_env_state() via run_ic() — restores physical state (attitude,
         rates, airspeed, alpha, beta).  The alpha/beta IC correction in
         utils.py ensures pitch and heading land correctly.
      2. All read-write JSBSim FDM properties copied directly on top —
         overrides engine state, FCS internals, trim settings, etc.
         These are R/O from JSBSim's perspective (can't set via IC) but
         ARE settable as live properties.
      3. Custom JSBSim properties not in the standard catalog (target state,
         error accumulators, integral errors) + Python wrapper state
         (pid_airspeed, action_hist, observation_deque).

    Returns a dict with copy statistics.
    """
    src_fdm = src_env.unwrapped.sim.fdm
    dst_fdm = dst_env.unwrapped.sim.fdm
    src_sim = src_env.unwrapped.sim
    dst_sim = dst_env.unwrapped.sim
    src_uw  = src_env.unwrapped
    dst_uw  = dst_env.unwrapped

    # 1. Physical state via run_ic() (attitude, rates, airspeed, alpha, beta) -
    set_env_state(dst_env, get_env_state(src_env))

    # 2. All RW FDM properties on top (engine, FCS, etc.) --------------------
    n_ok = n_fail = 0
    failed = []
    for entry in src_fdm.get_property_catalog():
        entry = entry.strip()
        if not entry.endswith('(RW)'):
            continue
        name = entry[:-5].strip()
        try:
            dst_fdm.set_property_value(name, src_fdm.get_property_value(name))
            n_ok += 1
        except Exception as e:
            failed.append((name, str(e)))
            n_fail += 1

    # 3. Custom JSBSim properties + Python wrapper state ---------------------
    for prop in [prp.target_roll_rad, prp.target_pitch_rad,
                 prp.roll_err,        prp.pitch_err,
                 prp.roll_integ_err,  prp.pitch_integ_err]:
        try:
            dst_sim[prop] = src_sim[prop]
        except Exception:
            pass

    if hasattr(src_uw, 'pid_airspeed') and hasattr(dst_uw, 'pid_airspeed'):
        dst_uw.pid_airspeed.integral   = src_uw.pid_airspeed.integral
        dst_uw.pid_airspeed.prev_error = src_uw.pid_airspeed.prev_error
        dst_uw.pid_airspeed.ref        = src_uw.pid_airspeed.ref

    if hasattr(src_uw, 'action_hist') and hasattr(dst_uw, 'action_hist'):
        dst_uw.action_hist.clear()
        for a in src_uw.action_hist:
            dst_uw.action_hist.append(a.copy() if hasattr(a, 'copy') else a)

    if hasattr(src_uw, 'observation_deque') and hasattr(dst_uw, 'observation_deque'):
        dst_uw.observation_deque.clear()
        for o in src_uw.observation_deque:
            dst_uw.observation_deque.append(o.copy() if hasattr(o, 'copy') else o)

    for attr in ('prev_target_roll', 'prev_target_pitch'):
        if hasattr(src_uw, attr):
            setattr(dst_uw, attr, getattr(src_uw, attr))

    return {'n_copied': n_ok, 'n_failed': n_fail, 'failed_props': failed}


# ============================================================================
# DETERMINISM TEST
# ============================================================================

def test_state_restore_determinism(n_warmup: int = 30, n_test: int = 15,
                                   target_roll: float = 55.0,
                                   target_pitch: float = 28.0) -> bool:
    """
    Verify that reset() + set_env_state() produces an env that is functionally
    identical to the original at state s0.

    Protocol
    --------
    1. Warm up main_env for n_warmup steps with fixed actions to reach s0.
    2. Save s0 = get_env_state(main_env).
    3. Create rollout_env, call reset() + set_env_state(s0) → clone at s0'.
    4. CHECK A: compare every field of s0 vs s0' (should be zero diff).
    5. Define a fixed action sequence of length n_test.
    6. Play it in main_env  → record obs[0..n_test] and final JSBSim state.
    7. Play it in rollout_env from s0' → record obs[0..n_test] and final JSBSim state.
    8. CHECK B: compare obs step-by-step and final JSBSim state (should be zero diff).

    Returns True if all checks pass within tolerance.
    """
    TOL = 1e-4
    passed = True

    print(f"\n{'='*70}")
    print("DETERMINISM TEST — reset()+set_env_state() clone fidelity")
    print(f"  Warm-up steps : {n_warmup}")
    print(f"  Test actions  : {n_test}")
    print(f"  Target        : roll={target_roll:+.1f}°  pitch={target_pitch:+.1f}°")
    print(f"{'='*70}")

    # ── Build both envs ───────────────────────────────────────────────────
    print("\n[1/5] Creating environments...")
    with suppress_output():
        main_env    = make_env()
        rollout_env = make_env()

    throttle_ref_main,    restore_main    = _throttle_patch(main_env)
    throttle_ref_rollout, restore_rollout = _throttle_patch(rollout_env)

    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)

    with suppress_output():
        main_env.reset(options={"fgear_target_roll": target_roll,
                                "fgear_target_pitch": target_pitch})
    main_env.unwrapped.set_target_state(np.array([target_roll_rad, target_pitch_rad]))

    # ── Warm-up: ramp aileron/elevator toward target, vary throttle ───────
    print(f"[2/5] Warming up main_env for {n_warmup} steps...")
    warmup_actions = []
    for i in range(n_warmup):
        t = i / max(n_warmup - 1, 1)
        ail = np.clip(0.3 * np.sin(2 * np.pi * t), -1.0, 1.0)
        ele = np.clip(-0.2 + 0.4 * t, -1.0, 1.0)
        thr = np.clip(0.3 + 0.2 * t, 0.0, 1.0)
        warmup_actions.append([ail, ele, thr])
        throttle_ref_main[0] = thr
        with suppress_output():
            main_env.step(np.array([ail, ele]))

    # ── Save s0 ───────────────────────────────────────────────────────────
    s0 = get_env_state(main_env)
    print(f"       s0 saved:  roll={np.rad2deg(s0['roll_rad']):+.3f}°  "
          f"pitch={np.rad2deg(s0['pitch_rad']):+.3f}°  "
          f"Va={s0['airspeed_kts']*1.852:.2f} kph  "
          f"q={s0['q_radps']:+.4f} rad/s")

    # ── Clone s0 into rollout_env ─────────────────────────────────────────
    print(f"\n[3/5] Cloning s0 into rollout_env via reset()+deep_clone_state()...")
    with suppress_output():
        rollout_env.reset()
    stats = deep_clone_state(main_env, rollout_env)
    print(f"       FDM properties copied: {stats['n_copied']}  failed: {stats['n_failed']}")
    if stats['failed_props']:
        print(f"       Failed: {[n for n,_ in stats['failed_props']]}")

    # ── CHECK A: state immediately after clone ────────────────────────────
    print(f"\n[CHECK A] s0 vs s0\' — physical state after clone (before any actions)")
    print(f"{'  Field':<22}  {'main':>12}  {'rollout':>12}  {'|diff|':>10}  result")
    print(f"  {'-'*70}")

    def _check(name, mv, rv):
        nonlocal passed
        diff = abs(mv - rv)
        ok   = diff <= TOL
        if not ok:
            passed = False
        status = "PASS" if ok else "FAIL ←"
        print(f"  {name:<22}  {mv:>12.6f}  {rv:>12.6f}  {diff:>10.2e}  {status}")

    sim_m = main_env.unwrapped.sim
    sim_r = rollout_env.unwrapped.sim
    _check("roll_rad",      sim_m[prp.roll_rad],      sim_r[prp.roll_rad])
    _check("pitch_rad",     sim_m[prp.pitch_rad],     sim_r[prp.pitch_rad])
    _check("heading_rad",   sim_m[prp.heading_rad],   sim_r[prp.heading_rad])
    _check("airspeed_kts",  sim_m[prp.airspeed_kts],  sim_r[prp.airspeed_kts])
    _check("p_radps",       sim_m[prp.p_radps],       sim_r[prp.p_radps])
    _check("q_radps",       sim_m[prp.q_radps],       sim_r[prp.q_radps])
    _check("r_radps",       sim_m[prp.r_radps],       sim_r[prp.r_radps])
    _check("altitude_ft",   sim_m[prp.altitude_sl_ft], sim_r[prp.altitude_sl_ft])
    _check("alpha_rad",     sim_m[prp.alpha_rad],     sim_r[prp.alpha_rad])
    _check("beta_rad",      sim_m[prp.beta_rad],      sim_r[prp.beta_rad])
    _check("aileron_cmd",   sim_m[prp.aileron_cmd],   sim_r[prp.aileron_cmd])
    _check("elevator_cmd",  sim_m[prp.elevator_cmd],  sim_r[prp.elevator_cmd])
    _check("throttle_cmd",  sim_m[prp.throttle_cmd],  sim_r[prp.throttle_cmd])

    # ── Define fixed test action sequence ─────────────────────────────────
    rng = np.random.default_rng(seed=0)
    test_actions = np.column_stack([
        rng.uniform(-0.8, 0.8,  n_test),   # aileron
        rng.uniform(-0.6, 0.6,  n_test),   # elevator
        rng.uniform( 0.2, 0.7,  n_test),   # throttle
    ])

    # ── Play test actions in main_env ─────────────────────────────────────
    print(f"\n[4/5] Playing {n_test} fixed actions in main_env...")
    main_obs_hist = []
    for k in range(n_test):
        throttle_ref_main[0] = test_actions[k, 2]
        with suppress_output():
            obs, _, term, trunc, _ = main_env.step(test_actions[k, :2])
        main_obs_hist.append(obs.copy())
        if term or trunc:
            print(f"       main_env terminated at step {k+1}!")
            break
    main_final = get_env_state(main_env)

    # ── Play same test actions in rollout_env from s0' ────────────────────
    print(f"       Playing {n_test} fixed actions in rollout_env from s0\'...")
    rollout_obs_hist = []
    for k in range(n_test):
        throttle_ref_rollout[0] = test_actions[k, 2]
        with suppress_output():
            obs, _, term, trunc, _ = rollout_env.step(test_actions[k, :2])
        rollout_obs_hist.append(obs.copy())
        if term or trunc:
            print(f"       rollout_env terminated at step {k+1}!")
            break

    rollout_final = get_env_state(rollout_env)

    # ── CHECK B: obs at every step ────────────────────────────────────────
    n_steps = min(len(main_obs_hist), len(rollout_obs_hist))
    obs_names = ["roll_rad", "pitch_rad", "airspeed_kph", "p_radps",
                 "q_radps", "r_radps", "roll_err", "pitch_err",
                 "alpha_rad", "beta_rad", "aileron_cmd", "elevator_cmd"]

    print(f"\n[CHECK B] obs trajectory divergence — step-by-step max |diff| per channel")
    print(f"  {'step':>5}  {'roll':>10}  {'pitch':>10}  {'Va':>10}  "
          f"{'p':>10}  {'q':>10}  {'r':>10}  result")
    print(f"  {'-'*80}")

    step_ok = True
    for k in range(n_steps):
        mo = main_obs_hist[k]
        ro = rollout_obs_hist[k]
        diffs = np.abs(mo - ro)
        ok = np.all(diffs[:6] <= TOL)
        if not ok:
            passed = False
            step_ok = False
        status = "ok" if ok else "FAIL ←"
        print(f"  {k+1:>5}  {diffs[0]:>10.2e}  {diffs[1]:>10.2e}  {diffs[2]:>10.2e}  "
              f"{diffs[3]:>10.2e}  {diffs[4]:>10.2e}  {diffs[5]:>10.2e}  {status}")

    if step_ok:
        print(f"  → All {n_steps} obs steps match within tolerance {TOL:.0e}")

    # ── CHECK B cont.: final JSBSim physical state ─────────────────────────
    print(f"\n  Final JSBSim physical state after {n_steps} actions:")
    print(f"  {'Field':<22}  {'main':>12}  {'rollout':>12}  {'|diff|':>10}  result")
    print(f"  {'-'*70}")
    for field in ['roll_rad', 'pitch_rad', 'airspeed_kts', 'p_radps',
                  'q_radps', 'r_radps', 'alpha_rad', 'beta_rad',
                  'aileron_cmd', 'elevator_cmd', 'throttle_cmd']:
        _check(field, main_final[field], rollout_final[field])

    # ── Cleanup & verdict ─────────────────────────────────────────────────
    restore_main()
    restore_rollout()
    main_env.close()
    rollout_env.close()

    print(f"\n{'='*70}")
    verdict = "PASS — envs are deterministically equivalent" if passed \
              else "FAIL — state diverges after clone (see FAIL rows above)"
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}\n")
    return passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MPPI with JSBSim as world model (ground-truth upper bound)'
    )
    parser.add_argument('--target-roll',      type=float, default=55.0)
    parser.add_argument('--target-pitch',     type=float, default=28.0)
    parser.add_argument('--steps',            type=int,   default=STEPS_20S,
                        help='Steps per run (default 2000 = 20 s)')
    parser.add_argument('--mppi-samples',     type=int,   default=100)
    parser.add_argument('--mppi-horizon',     type=int,   default=20)
    parser.add_argument('--mppi-temperature', type=float, default=0.5)
    parser.add_argument('--mppi-noise-std',   type=float, default=0.5)
    parser.add_argument('--mppi-va-weight',   type=float, default=1)
    parser.add_argument('--mppi-iters',       type=int,   default=None,
                        help='Iterative MPPI: number of refinement passes per timestep (TD-MPC style). '
                             'Omit for vanilla single-pass MPPI.')
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--render-mode',      type=str,   default='none',
                        choices=['none', 'plot_anim', 'plot_end', 'ext_log', 'fgear', 'fgear_plot'],
                        help='Visualization mode for the main env (default: none). '
                             'plot_anim requires: sudo apt install python3-tk')
    parser.add_argument('--test-determinism', action='store_true',
                        help='Run the state-restore determinism test and exit')
    parser.add_argument('--test-warmup',      type=int,   default=30,
                        help='Warm-up steps for --test-determinism (default: 30)')
    parser.add_argument('--test-actions',     type=int,   default=15,
                        help='Fixed test actions for --test-determinism (default: 15)')
    args = parser.parse_args()

    if args.test_determinism:
        test_state_restore_determinism(
            n_warmup=args.test_warmup,
            n_test=args.test_actions,
            target_roll=args.target_roll,
            target_pitch=args.target_pitch,
        )
        return

    mppi_cfg = dict(
        samples=args.mppi_samples, horizon=args.mppi_horizon,
        temperature=args.mppi_temperature, noise_std=args.mppi_noise_std,
        va_weight=args.mppi_va_weight, iters=args.mppi_iters,
    )
    tag = f"r{args.target_roll:.0f}_p{args.target_pitch:.0f}"

    print(f"\nMPPI Env-Model Run")
    print(f"  Target : Roll={args.target_roll:+.1f}°  Pitch={args.target_pitch:+.1f}°")
    print(f"  Steps  : {args.steps} ({args.steps * DT:.0f} s)")
    print(f"  MPPI   : samples={args.mppi_samples}  horizon={args.mppi_horizon}  "
          f"temp={args.mppi_temperature}  noise={args.mppi_noise_std}")
    print(f"  Seed   : {args.seed}")
    print(f"  Render : {args.render_mode}")

    result = run_mppi_env(
        args.target_roll, args.target_pitch, args.steps, mppi_cfg, args.seed,
        render_mode=args.render_mode,
    )

    print(f"\nSaving plots to {SAVE_DIR}/")
    fname = f"{_safe_label(result['label'])}_{tag}_plots.png"
    plot_model_result(result, args.target_roll, args.target_pitch, SAVE_DIR / fname,
                      target_va=TARGET_VA_KPH)

    save_metrics([result], args.target_roll, args.target_pitch,
                 SAVE_DIR / f"metrics_{tag}.csv")


if __name__ == '__main__':
    main()
