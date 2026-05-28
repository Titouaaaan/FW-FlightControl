#!/usr/bin/env python3
"""
MPPI Oracle — JSBSim as World Model
=====================================
Uses JSBSim itself as the dynamics model for MPPI rollouts.
This is the theoretical upper bound: the world model IS the simulator.

Clone strategy
--------------
Each rollout needs an independent copy of the simulator at its current state.
The hard problem: JSBSim's Adams–Bashforth 4 integrator maintains internal
history deques (dqUVWidot, dqPQRidot) that are NOT accessible via the property
interface — so property-based cloning always produces a deque mismatch that
makes the first integration step behave like Euler.

Solution: os.fork().  Forking copies the entire process memory including all
C++ internals.  The child inherits a bit-perfect snapshot → zero divergence.
Verified via --sanity: every obs channel is exactly 0.000e+00 across H=20 steps.

Fallback (--use-replay, non-Linux):
  Maintains a 4-entry buffer of (snapshot, action) pairs.  Each rollout
  restores from 4 steps ago then replays those 4 actions, rebuilding the
  deque.  p/r error ~30× better than raw deep_clone; small Va offset (~0.22 kph)
  that is action-independent and cancels in MPPI's softmax.

Usage
-----
    python mppi_env_only.py --sanity                    # clone sanity check
    python mppi_env_only.py                             # run oracle (fork)
    python mppi_env_only.py --use-replay                # replay fallback
    python mppi_env_only.py --steps 2000 --mppi-samples 100
"""

import os, sys, struct, time, pickle, argparse
import numpy as np
from collections import deque as Deque
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import fw_jsbgym                                           # registers gym envs
import gymnasium as gym
from fw_jsbgym.utils import jsbsim_properties as prp

from fw_flightcontrol.physics.utils import (
    get_env_state, set_env_state,
    suppress_output, _throttle_patch, _safe_label,
    plot_model_result, save_metrics, compute_convergence_stats,
)
from fw_flightcontrol.physics.tests.mppi_test import MPPIController


# ── Paths & constants ──────────────────────────────────────────────────────────
_FC_DIR       = Path(__file__).parent.parent.parent
CONFIG_DIR    = str(_FC_DIR / 'config')
NOATMO_YAML   = str(_FC_DIR / 'config' / 'env' / 'jsbsim' / 'noatmo.yaml')
SAVE_DIR      = _FC_DIR / 'data' / 'oracle_mppi'

DT            = 0.01      # s per simulation step
TARGET_VA_KPH = 60.0      # cruise airspeed target [kph]
STEPS_20S     = 2000      # 20 s run


# ── Environment ────────────────────────────────────────────────────────────────

def make_env(render_mode: str = 'none', telemetry_file: str = '') -> gym.Env:
    """Create and initialise one JSBSim gymnasium environment."""
    try:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name='default')
    except Exception:
        cfg = OmegaConf.create({'env': {}})
    cfg.env.jsbsim = OmegaConf.load(NOATMO_YAML)
    if render_mode != 'none' and not telemetry_file:
        os.makedirs('telemetry', exist_ok=True)
        telemetry_file = 'telemetry/telemetry.csv'
    env = gym.make('ACBohnNoVaIErr-v0', cfg_env=cfg.env,
                   render_mode=render_mode, telemetry_file=telemetry_file)
    env.unwrapped.init()
    return env


# ── Rollout: fork (primary, Linux/WSL2) ───────────────────────────────────────

def _rollout_fork(main_env, thr_ref, actions: np.ndarray,
                  roll_rad: float, pitch_rad: float,
                  va_kph: float, va_weight: float) -> np.ndarray:
    """Run N rollouts via os.fork() — bit-perfect clone of all JSBSim state.

    For each sample the child process inherits an exact copy of main_env
    (including the AB4 deque history), runs H steps, and returns a scalar
    cost.  main_env in the parent is never modified.
    """
    n, horizon, _ = actions.shape
    costs = np.zeros(n)
    for k in range(n):
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:                                   # ── child ──
            os.close(r_fd)
            cost = 0.0
            for h in range(horizon):
                thr_ref[0] = float(actions[k, h, 2])
                with suppress_output():
                    obs, _, term, trunc, _ = main_env.step(actions[k, h, :2])
                cost += (abs(obs[0] - roll_rad)
                         + abs(obs[1] - pitch_rad)
                         + va_weight * abs(obs[2] - va_kph))
                if term or trunc:
                    cost += 100.0
                    break
            os.write(w_fd, struct.pack('d', cost))
            os.close(w_fd)
            os._exit(0)
        os.close(w_fd)                                 # ── parent ──
        os.waitpid(pid, 0)
        costs[k] = struct.unpack('d', os.read(r_fd, 8))[0]
        os.close(r_fd)
    return costs


# ── Rollout: deep-clone + optional replay (fallback) ──────────────────────────
# Property-based clone cannot access the AB4 deques, so the first step after
# cloning is Euler-like.  The 4-step replay partially rebuilds the deque.

_IC_VEL_SKIP = {
    'ic/vn-fps', 'ic/ve-fps', 'ic/vd-fps',
    'ic/u-fps',  'ic/v-fps',  'ic/w-fps',
    'ic/vt-fps', 'ic/vt-kts', 'ic/vc-fps', 'ic/vc-kts',
    'ic/vg-fps', 'ic/vg-kts', 'ic/ve-kts', 'ic/mach',
}


def deep_clone_state(src_env, dst_env) -> None:
    """Clone JSBSim state via run_ic() + all read-write properties.

    Note: AB4 deques are inaccessible through the property interface and will
    be wrong.  Use os.fork() for a guaranteed bit-perfect copy.
    """
    src_fdm = src_env.unwrapped.sim.fdm
    dst_fdm = dst_env.unwrapped.sim.fdm
    src_sim = src_env.unwrapped.sim
    dst_sim = dst_env.unwrapped.sim
    src_uw  = src_env.unwrapped
    dst_uw  = dst_env.unwrapped
    s0      = get_env_state(src_env)

    def _copy_rw():
        for entry in src_fdm.get_property_catalog():
            entry = entry.strip()
            if not entry.endswith('(RW)'):
                continue
            name = entry[:-5].strip()
            if name in _IC_VEL_SKIP:
                continue
            try:
                dst_fdm.set_property_value(name, src_fdm.get_property_value(name))
            except Exception:
                pass

    def _apply_ic():
        dst_sim[prp.ic_roll_rad]     = s0['roll_rad']
        dst_sim[prp.ic_pitch_rad]    = s0['pitch_rad']   - s0['alpha_rad']
        dst_sim[prp.ic_heading_rad]  = s0['heading_rad'] + s0['beta_rad']
        dst_sim[prp.ic_airspeed_kts] = s0['airspeed_kts']
        dst_sim[prp.ic_p_radps]      = s0['p_radps']
        dst_sim[prp.ic_q_radps]      = s0['q_radps']
        dst_sim[prp.ic_r_radps]      = s0['r_radps']
        dst_sim[prp.ic_altitude_ft]  = s0['altitude_ft']
        dst_fdm['ic/alpha-rad']      = s0['alpha_rad']
        dst_fdm['ic/beta-rad']       = s0['beta_rad']
        dst_sim[prp.throttle_cmd]    = s0['throttle']
        dst_sim[prp.aileron_cmd]     = s0['aileron_cmd']
        dst_sim[prp.elevator_cmd]    = s0['elevator_cmd']
        dst_fdm['fcs/left-aileron-pos-rad']  = s0.get('fcs_aileron_pos_rad', 0.0)
        dst_fdm['fcs/right-aileron-pos-rad'] = s0.get('fcs_aileron_pos_rad', 0.0)
        dst_fdm['fcs/elevator-pos-rad']      = s0.get('fcs_elevator_pos_rad', 0.0)
        dst_fdm['fcs/throttle-pos-norm']     = s0.get('fcs_throttle_pos_norm', s0['throttle'])

    set_env_state(dst_env, s0)
    _copy_rw(); _apply_ic(); dst_fdm.run_ic()
    dst_sim[prp.throttle_cmd] = s0['throttle']
    dst_sim[prp.aileron_cmd]  = s0['aileron_cmd']
    dst_sim[prp.elevator_cmd] = s0['elevator_cmd']
    _copy_rw()

    for prop in (prp.target_roll_rad, prp.target_pitch_rad,
                 prp.roll_err, prp.pitch_err,
                 prp.roll_integ_err, prp.pitch_integ_err):
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


def _save_snapshot(env) -> dict:
    """Capture the complete env state into a dict (for replay mode)."""
    fdm = env.unwrapped.sim.fdm
    sim = env.unwrapped.sim
    uw  = env.unwrapped
    snap = get_env_state(env)
    rw = {}
    for entry in fdm.get_property_catalog():
        entry = entry.strip()
        if not entry.endswith('(RW)'):
            continue
        name = entry[:-5].strip()
        if name in _IC_VEL_SKIP:
            continue
        try:
            rw[name] = fdm.get_property_value(name)
        except Exception:
            pass
    snap['_rw'] = rw
    for prop, key in [(prp.target_roll_rad, '_tgt_roll'),
                      (prp.target_pitch_rad, '_tgt_pitch'),
                      (prp.roll_err, '_rerr'), (prp.pitch_err, '_perr'),
                      (prp.roll_integ_err, '_rint'), (prp.pitch_integ_err, '_pint')]:
        try:
            snap[key] = float(sim[prop])
        except Exception:
            snap[key] = 0.0
    if hasattr(uw, 'pid_airspeed'):
        snap['_pid'] = {'integral': uw.pid_airspeed.integral,
                        'prev_error': uw.pid_airspeed.prev_error,
                        'ref': uw.pid_airspeed.ref}
    if hasattr(uw, 'action_hist'):
        snap['_act'] = [a.copy() if hasattr(a, 'copy') else a for a in uw.action_hist]
    if hasattr(uw, 'observation_deque'):
        snap['_obs'] = [o.copy() if hasattr(o, 'copy') else o for o in uw.observation_deque]
    return snap


def _restore_snapshot(dst_env, snap: dict) -> None:
    """Restore dst_env from a snapshot dict (double run_ic protocol)."""
    dst_fdm = dst_env.unwrapped.sim.fdm
    dst_sim = dst_env.unwrapped.sim
    dst_uw  = dst_env.unwrapped

    def _apply_rw():
        for name, val in snap.get('_rw', {}).items():
            try:
                dst_fdm.set_property_value(name, val)
            except Exception:
                pass

    def _apply_ic():
        dst_sim[prp.ic_roll_rad]     = snap['roll_rad']
        dst_sim[prp.ic_pitch_rad]    = snap['pitch_rad']   - snap['alpha_rad']
        dst_sim[prp.ic_heading_rad]  = snap['heading_rad'] + snap['beta_rad']
        dst_sim[prp.ic_airspeed_kts] = snap['airspeed_kts']
        dst_sim[prp.ic_p_radps]      = snap['p_radps']
        dst_sim[prp.ic_q_radps]      = snap['q_radps']
        dst_sim[prp.ic_r_radps]      = snap['r_radps']
        dst_sim[prp.ic_altitude_ft]  = snap['altitude_ft']
        dst_fdm['ic/alpha-rad']      = snap['alpha_rad']
        dst_fdm['ic/beta-rad']       = snap['beta_rad']
        dst_sim[prp.throttle_cmd]    = snap['throttle']
        dst_sim[prp.aileron_cmd]     = snap['aileron_cmd']
        dst_sim[prp.elevator_cmd]    = snap['elevator_cmd']
        dst_fdm['fcs/left-aileron-pos-rad']  = snap.get('fcs_aileron_pos_rad', 0.0)
        dst_fdm['fcs/right-aileron-pos-rad'] = snap.get('fcs_aileron_pos_rad', 0.0)
        dst_fdm['fcs/elevator-pos-rad']      = snap.get('fcs_elevator_pos_rad', 0.0)
        dst_fdm['fcs/throttle-pos-norm']     = snap.get('fcs_throttle_pos_norm', snap['throttle'])

    set_env_state(dst_env, snap); _apply_rw(); _apply_ic(); dst_fdm.run_ic()
    dst_sim[prp.throttle_cmd] = snap['throttle']
    dst_sim[prp.aileron_cmd]  = snap['aileron_cmd']
    dst_sim[prp.elevator_cmd] = snap['elevator_cmd']
    _apply_rw()
    for prop, key in [(prp.target_roll_rad, '_tgt_roll'),
                      (prp.target_pitch_rad, '_tgt_pitch'),
                      (prp.roll_err, '_rerr'), (prp.pitch_err, '_perr'),
                      (prp.roll_integ_err, '_rint'), (prp.pitch_integ_err, '_pint')]:
        if key in snap:
            try:
                dst_sim[prop] = snap[key]
            except Exception:
                pass
    if hasattr(dst_uw, 'pid_airspeed') and '_pid' in snap:
        dst_uw.pid_airspeed.integral   = snap['_pid']['integral']
        dst_uw.pid_airspeed.prev_error = snap['_pid']['prev_error']
        dst_uw.pid_airspeed.ref        = snap['_pid']['ref']
    if hasattr(dst_uw, 'action_hist') and '_act' in snap:
        dst_uw.action_hist.clear()
        for a in snap['_act']:
            dst_uw.action_hist.append(a.copy() if hasattr(a, 'copy') else a)
    if hasattr(dst_uw, 'observation_deque') and '_obs' in snap:
        dst_uw.observation_deque.clear()
        for o in snap['_obs']:
            dst_uw.observation_deque.append(o.copy() if hasattr(o, 'copy') else o)


def _rollout_clone(rollout_env, main_env, actions: np.ndarray,
                   roll_rad: float, pitch_rad: float,
                   va_kph: float, va_weight: float,
                   replay_buf: Optional[Deque] = None) -> np.ndarray:
    """Run N rollouts via deep_clone_state or 4-step replay (fallback)."""
    n, horizon, _ = actions.shape
    costs = np.zeros(n)
    thr_ref, restore = _throttle_patch(rollout_env)
    use_replay = replay_buf is not None and len(replay_buf) >= 4

    for k in range(n):
        with suppress_output():
            rollout_env.reset()
        if use_replay:
            with suppress_output():
                _restore_snapshot(rollout_env, replay_buf[0][0])
            for _, past_act in replay_buf:
                thr_ref[0] = float(past_act[2])
                with suppress_output():
                    rollout_env.step(past_act[:2])
        else:
            with suppress_output():
                deep_clone_state(main_env, rollout_env)

        cost = 0.0
        for h in range(horizon):
            thr_ref[0] = float(actions[k, h, 2])
            obs, _, term, trunc, _ = rollout_env.step(actions[k, h, :2])
            cost += (abs(obs[0] - roll_rad)
                     + abs(obs[1] - pitch_rad)
                     + va_weight * abs(obs[2] - va_kph))
            if term or trunc:
                cost += 100.0
                break
        costs[k] = cost

    restore()
    return costs


# ── MPPI loop ──────────────────────────────────────────────────────────────────

def run_mppi_env(target_roll: float, target_pitch: float, max_steps: int,
                 mppi_cfg: dict, seed: int,
                 render_mode: str = 'none',
                 use_fork: bool = True,
                 use_replay: bool = False,
                 log_every: int = 100) -> dict:
    """Run the MPPI oracle controller.

    Args:
        target_roll/pitch : attitude setpoints [°]
        max_steps         : total simulation steps (1 step = DT seconds)
        mppi_cfg          : samples, horizon, temperature, noise_std, va_weight,
                            iters (all keys)
        seed              : numpy RNG seed for reproducibility
        render_mode       : JSBSim render mode
        use_fork          : os.fork() clone — bit-perfect, Linux/WSL2 only
        use_replay        : 4-step replay clone — fallback, any OS
        log_every         : print a progress row every N steps (0 = silent)

    Returns a dict with roll/pitch/va/p/q/r histories, actions, step_times,
    steps, terminated — compatible with plot_model_result / save_metrics.
    """
    with suppress_output():
        main_env    = make_env(render_mode=render_mode,
                               telemetry_file='telemetry/mppi_env.csv')
        rollout_env = None if use_fork else make_env()

    thr_ref, restore_patch = _throttle_patch(main_env)
    controller = MPPIController(
        horizon=mppi_cfg['horizon'], action_dim=3,
        num_samples=mppi_cfg['samples'],
        temperature=mppi_cfg['temperature'],
        noise_std=mppi_cfg['noise_std'],
    )
    np.random.seed(seed)
    controller.reset()

    with suppress_output():
        obs, _ = main_env.reset(options={"fgear_target_roll":  target_roll,
                                          "fgear_target_pitch": target_pitch})

    roll_rad  = np.deg2rad(target_roll)
    pitch_rad = np.deg2rad(target_pitch)
    va_weight = mppi_cfg.get('va_weight', 0.1)
    n_iters   = mppi_cfg.get('iters') or 1
    main_env.unwrapped.set_target_state(np.array([roll_rad, pitch_rad]))

    replay_buf: Deque = Deque(maxlen=4)   # used only when use_replay=True

    roll_h, pitch_h, va_h, p_h, q_h, r_h = [], [], [], [], [], []
    act_h, times = [], []
    terminated = False

    clone_mode = 'fork' if use_fork else ('replay' if use_replay else 'deep_clone')

    # ── Header (printed once, above the tqdm bar) ─────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  Oracle MPPI | roll={target_roll:+.1f}° pitch={target_pitch:+.1f}° "
          f"Va={TARGET_VA_KPH:.0f} kph | clone={clone_mode}")
    print(f"  N={mppi_cfg['samples']} H={mppi_cfg['horizon']} "
          f"λ={mppi_cfg['temperature']} σ={mppi_cfg['noise_std']} "
          f"va_w={va_weight} seed={seed}")
    print(f"{'─'*68}")
    if log_every > 0:
        print(f"  {'step':>5}  {'roll(°)':>8}  {'err_r(°)':>9}  "
              f"{'pitch(°)':>9}  {'err_p(°)':>9}  {'Va(kph)':>8}  {'ms':>6}")
        print(f"  {'─'*63}")

    with tqdm(total=max_steps, desc='MPPI', unit='step', leave=True) as pbar:
        for step in range(max_steps):
            t0   = time.time()
            best = np.zeros(3)

            rbuf_arg = replay_buf if (use_replay and len(replay_buf) >= 4) else None

            for _ in range(n_iters):
                sampled = controller.sample_actions()
                costs = (_rollout_fork(main_env, thr_ref, sampled,
                                       roll_rad, pitch_rad, TARGET_VA_KPH, va_weight)
                         if use_fork else
                         _rollout_clone(rollout_env, main_env, sampled,
                                        roll_rad, pitch_rad, TARGET_VA_KPH, va_weight,
                                        rbuf_arg))
                valid = np.isfinite(costs)
                if not valid.any():
                    continue
                cs = np.where(valid, costs, np.nanmax(costs[valid]) * 10.0)
                w  = np.exp(-(cs - cs.min()) / controller.temperature)
                w /= w.sum()
                controller.mean_actions = np.einsum('k,khd->hd', w, sampled)
                best  = controller.mean_actions[0].copy()
                best[:2] = np.clip(best[:2], -1.0, 1.0)
                best[2]  = np.clip(best[2],  0.0,  1.0)

            controller._shift_mean()

            if use_replay:
                replay_buf.append((_save_snapshot(main_env), best.copy()))

            thr_ref[0] = float(best[2])
            obs, _, term, trunc, _ = main_env.step(best[:2])
            dt_ms = (time.time() - t0) * 1000

            r_deg, p_deg = np.rad2deg(obs[0]), np.rad2deg(obs[1])
            roll_h.append(r_deg);  pitch_h.append(p_deg);  va_h.append(obs[2])
            p_h.append(obs[3]);    q_h.append(obs[4]);      r_h.append(obs[5])
            act_h.append(best.copy()); times.append(time.time() - t0)

            # Update live tqdm bar with current errors
            pbar.set_postfix(
                roll=f'{r_deg:+.1f}°',
                err_r=f'{abs(r_deg-target_roll):.1f}°',
                pitch=f'{p_deg:+.1f}°',
                err_p=f'{abs(p_deg-target_pitch):.1f}°',
                Va=f'{obs[2]:.1f}',
                ms=f'{dt_ms:.0f}',
            )
            pbar.update(1)

            # Periodic detailed row above the bar
            if log_every > 0 and (step % log_every == 0 or step == max_steps - 1):
                pbar.write(
                    f"  {step+1:>5}  {r_deg:>+8.2f}  "
                    f"{abs(r_deg - target_roll):>9.2f}  "
                    f"{p_deg:>+9.2f}  "
                    f"{abs(p_deg - target_pitch):>9.2f}  "
                    f"{obs[2]:>8.2f}  {dt_ms:>6.0f}"
                )

            if term or trunc:
                terminated = True
                break

    restore_patch()
    main_env.close()
    if rollout_env is not None:
        rollout_env.close()

    # ── End-of-run summary ─────────────────────────────────────────────────────
    if times:
        ra = np.array(roll_h);  pa = np.array(pitch_h)
        rc = compute_convergence_stats(roll_h,  target_roll)
        pc = compute_convergence_stats(pitch_h, target_pitch)
        n  = len(times)
        print(f"\n  {'─'*63}")
        print(f"  Summary  ({n} steps, {sum(times):.0f} s wall-clock)")
        r_conv = f"{rc['convergence_step']*DT:.1f} s" if rc['convergence_step'] else 'never'
        p_conv = f"{pc['convergence_step']*DT:.1f} s" if pc['convergence_step'] else 'never'
        print(f"  Roll  : mean err={np.mean(np.abs(ra-target_roll)):.2f}°  "
              f"converge={r_conv}  "
              f"ss={rc['steady_mean_error']:.2f}°±{rc['steady_std']:.2f}°")
        print(f"  Pitch : mean err={np.mean(np.abs(pa-target_pitch)):.2f}°  "
              f"converge={p_conv}  "
              f"ss={pc['steady_mean_error']:.2f}°±{pc['steady_std']:.2f}°")
        print(f"  Step  : mean={np.mean(times)*1000:.0f} ms  "
              f"min={np.min(times)*1000:.0f} ms  max={np.max(times)*1000:.0f} ms")
        print(f"  {'─'*63}\n")

    return dict(label=f'JSBSim_{clone_mode}',
                roll=roll_h, pitch=pitch_h, va=va_h,
                p=p_h, q=q_h, r=r_h,
                actions=np.array(act_h), step_times=times,
                steps=len(roll_h), terminated=terminated)


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(n_warmup: int = 50, n_test: int = 20,
                 target_roll: float = 55.0, target_pitch: float = 28.0) -> bool:
    """Verify that os.fork() produces a bit-perfect clone at a mid-run state.

    Protocol:
      1. Warm up one env for n_warmup steps (builds non-trivial deque history).
      2. os.fork(): child and parent each run the same n_test actions.
      3. Compare obs step by step — fork diff must be exactly zero.
      4. Also run deep_clone_state for contrast, showing the AB4 deque error.

    Prints a full step-by-step comparison table and timing.
    Returns True if fork diff is identically zero.
    """
    t0 = time.time()
    print(f"\n{'═'*65}")
    print(f"  SANITY CHECK — os.fork() clone fidelity at mid-run state")
    print(f"  Warm-up: {n_warmup} steps    Test seq: {n_test} actions    seed=0")
    print(f"{'═'*65}")

    target_r = np.deg2rad(target_roll)
    target_p = np.deg2rad(target_pitch)

    with suppress_output():
        env_main  = make_env()
        env_clone = make_env()        # only for deep_clone comparison

    thr_main,  _ = _throttle_patch(env_main)
    thr_clone, _ = _throttle_patch(env_clone)

    with suppress_output():
        env_main.reset(options={"fgear_target_roll":  target_roll,
                                "fgear_target_pitch": target_pitch})
    env_main.unwrapped.set_target_state(np.array([target_r, target_p]))

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"\n  [1/3] Warming up {n_warmup} steps ...", end='', flush=True)
    t_wu = time.time()
    for i in range(n_warmup):
        t   = i / max(n_warmup - 1, 1)
        ail = float(np.clip(0.3 * np.sin(2 * np.pi * t), -1, 1))
        ele = float(np.clip(-0.2 + 0.4 * t, -1, 1))
        thr = float(np.clip(0.3 + 0.2 * t, 0, 1))
        thr_main[0] = thr
        with suppress_output():
            env_main.step(np.array([ail, ele]))
    sim = env_main.unwrapped.sim
    print(f" {time.time()-t_wu:.2f} s")
    print(f"        State at s{n_warmup}: "
          f"roll={np.rad2deg(sim[prp.roll_rad]):+.2f}°  "
          f"pitch={np.rad2deg(sim[prp.pitch_rad]):+.2f}°  "
          f"Va={sim[prp.airspeed_kts]*1.852:.2f} kph  "
          f"p={sim[prp.p_radps]:+.4f} rad/s")

    # ── Test actions ──────────────────────────────────────────────────────────
    rng  = np.random.default_rng(seed=0)
    acts = np.column_stack([rng.uniform(-0.8, 0.8, n_test),
                             rng.uniform(-0.6, 0.6, n_test),
                             rng.uniform( 0.2, 0.7, n_test)])

    # ── os.fork() ─────────────────────────────────────────────────────────────
    print(f"\n  [2/3] os.fork() rollout ...", end='', flush=True)
    t_fork = time.time()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # child
        os.close(r_fd)
        child_obs = []
        for k in range(n_test):
            thr_main[0] = float(acts[k, 2])
            with suppress_output():
                obs, *_ = env_main.step(acts[k, :2])
            child_obs.append(obs.copy())
        data = pickle.dumps(child_obs)
        sent = 0
        while sent < len(data):
            sent += os.write(w_fd, data[sent:sent+65536])
        os.close(w_fd)
        os._exit(0)

    os.close(w_fd)
    parent_obs = []
    for k in range(n_test):                         # parent = ground truth
        thr_main[0] = float(acts[k, 2])
        with suppress_output():
            obs, *_ = env_main.step(acts[k, :2])
        parent_obs.append(obs.copy())

    os.waitpid(pid, 0)
    chunks = []
    while True:
        c = os.read(r_fd, 65536)
        if not c: break
        chunks.append(c)
    os.close(r_fd)
    fork_obs = pickle.loads(b''.join(chunks))
    t_fork_done = time.time() - t_fork
    print(f" {t_fork_done:.2f} s  (≈{t_fork_done/n_test*1000:.0f} ms/sample)")

    # ── deep_clone_state comparison ───────────────────────────────────────────
    print(f"  [3/3] deep_clone_state rollout ...", end='', flush=True)
    t_dc = time.time()
    with suppress_output():
        env_clone.reset()
    deep_clone_state(env_main, env_clone)
    clone_obs = []
    for k in range(n_test):
        thr_clone[0] = float(acts[k, 2])
        with suppress_output():
            obs, *_ = env_clone.step(acts[k, :2])
        clone_obs.append(obs.copy())
    print(f" {time.time()-t_dc:.2f} s")
    env_main.close(); env_clone.close()

    # ── Print table ───────────────────────────────────────────────────────────
    OBS   = ['roll(rad)', 'pitch(rad)', 'Va(kph)', 'p(rad/s)', 'r(rad/s)']
    IDX   = [0, 1, 2, 3, 5]
    W     = 10
    hdr   = "  ".join(f"{n:^{W}}" for n in OBS)
    sep   = "─" * (8 + len(OBS) * (W + 2))

    fork_d  = [np.abs(np.array(parent_obs[k]) - np.array(fork_obs[k]))  for k in range(n_test)]
    clone_d = [np.abs(np.array(parent_obs[k]) - np.array(clone_obs[k])) for k in range(n_test)]

    print(f"\n  ── os.fork() | diff vs ground truth (expect all zeros) ──")
    print(f"  {'step':>4}  {hdr}")
    print(f"  {sep}")
    for k in range(n_test):
        row = "  ".join(f"{fork_d[k][i]:>{W}.2e}" for i in IDX)
        print(f"  {k+1:>4}  {row}")
    fork_max = max(fork_d[k][i] for k in range(n_test) for i in IDX)
    print(f"  MAX over all channels: {fork_max:.2e}")

    print(f"\n  ── deep_clone_state | diff vs ground truth (AB4 deque error) ──")
    print(f"  {'step':>4}  {hdr}")
    print(f"  {sep}")
    for k in range(n_test):
        row = "  ".join(f"{clone_d[k][i]:>{W}.2e}" for i in IDX)
        print(f"  {k+1:>4}  {row}")
    cmax_va = max(clone_d[k][2] for k in range(n_test))
    cmax_p  = max(clone_d[k][3] for k in range(n_test))
    print(f"  MAX: Va={cmax_va:.2e} kph  p={cmax_p:.2e} rad/s")

    # ── Verdict ───────────────────────────────────────────────────────────────
    passed = fork_max == 0.0
    print(f"\n{'═'*65}")
    if passed:
        print(f"  RESULT: PASS — os.fork() is bit-perfect (max diff = 0.0)")
    else:
        print(f"  RESULT: FAIL — unexpected divergence in fork clone ({fork_max:.2e})")
    print(f"  Total time: {time.time()-t0:.1f} s")
    print(f"{'═'*65}\n")
    return passed


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MPPI Oracle — JSBSim as world model')
    parser.add_argument('--target-roll',      type=float, default=55.0)
    parser.add_argument('--target-pitch',     type=float, default=28.0)
    parser.add_argument('--steps',            type=int,   default=STEPS_20S)
    parser.add_argument('--mppi-samples',     type=int,   default=256)
    parser.add_argument('--mppi-horizon',     type=int,   default=40)
    parser.add_argument('--mppi-temperature', type=float, default=0.3)
    parser.add_argument('--mppi-noise-std',   type=float, default=0.5)
    parser.add_argument('--mppi-va-weight',   type=float, default=1)
    parser.add_argument('--mppi-iters',       type=int,   default=None,
                        help='MPPI refinement passes per step (default: 1)')
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--render-mode',      type=str,   default='none',
                        choices=['none', 'plot_anim', 'plot_end',
                                 'ext_log', 'fgear', 'fgear_plot'])
    parser.add_argument('--use-fork',         action='store_true',
                        help='os.fork() cloning — bit-perfect (Linux/WSL2)')
    parser.add_argument('--use-replay',       action='store_true',
                        help='4-step replay cloning — fallback, any OS')
    parser.add_argument('--log-every',        type=int,   default=1,
                        help='Print progress row every N steps (0=silent)')
    parser.add_argument('--sanity',           action='store_true',
                        help='Run clone sanity check and exit')
    parser.add_argument('--sanity-warmup',    type=int,   default=50)
    parser.add_argument('--sanity-steps',     type=int,   default=20)
    args = parser.parse_args()

    if args.sanity:
        sanity_check(args.sanity_warmup, args.sanity_steps,
                     args.target_roll, args.target_pitch)
        return

    mppi_cfg = dict(
        samples=args.mppi_samples, horizon=args.mppi_horizon,
        temperature=args.mppi_temperature, noise_std=args.mppi_noise_std,
        va_weight=args.mppi_va_weight, iters=args.mppi_iters,
    )

    result = run_mppi_env(
        args.target_roll, args.target_pitch, args.steps,
        mppi_cfg, args.seed,
        render_mode=args.render_mode,
        use_fork=args.use_fork,
        use_replay=args.use_replay,
        log_every=args.log_every,
    )

    tag  = f"r{args.target_roll:.0f}_p{args.target_pitch:.0f}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_result(result, args.target_roll, args.target_pitch,
                      SAVE_DIR / f"{_safe_label(result['label'])}_{tag}.png",
                      target_va=TARGET_VA_KPH)
    save_metrics([result], args.target_roll, args.target_pitch,
                 SAVE_DIR / f"metrics_{tag}_{_safe_label(result['label'])}.csv")


if __name__ == '__main__':
    main()
