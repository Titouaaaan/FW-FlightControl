#!/usr/bin/env python3
import os, sys, time, pickle, argparse
import numpy as np
from pathlib import Path

from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import fw_jsbgym
import gymnasium as gym
from fw_jsbgym.utils import jsbsim_properties as prp

from fw_flightcontrol.physics.utils import (
    suppress_output, _throttle_patch, _safe_label,
    plot_model_result, save_metrics, compute_convergence_stats,
)
from fw_flightcontrol.physics.mppi import MPPIController, compute_costs


# ── Paths & constants ──────────────────────────────────────────────────────────
_FC_DIR       = Path(__file__).parent.parent.parent
CONFIG_DIR    = str(_FC_DIR / 'config')
NOATMO_YAML   = str(_FC_DIR / 'config' / 'env' / 'jsbsim' / 'noatmo.yaml')
SAVE_DIR      = Path(__file__).parent.parent / 'data' / 'plots' / 'oracle_mppi_topk'

DT            = 0.01
TARGET_VA_KPH = 60.0

# ── Environment ────────────────────────────────────────────────────────────────

def make_env(render_mode: str = 'none', telemetry_file: str = '') -> gym.Env:
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


# ── Fork rollout ───────────────────────────────────────────────────────────────

def _rollout_fork(main_env, thr_ref, actions: np.ndarray) -> np.ndarray:
    """Run N rollouts in parallel via os.fork(). All children are launched
    before any waitpid, so they run concurrently up to the OS scheduler limit.
    Each child redirects output once at startup (not per step).

    Returns:
        trajectories: (N, H, 3) — [roll_rad, pitch_rad, Va_ms].
                      Steps after early termination are NaN.
    """
    n, horizon, _ = actions.shape
    bytes_per_traj = horizon * 3 * 8          # H × 3 channels × 8 bytes
    pids_and_pipes = []

    for k in range(n):
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:                                   # ── child ──
            os.close(r_fd)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
            traj = np.full((horizon, 3), np.nan)
            for h in range(horizon):
                thr_ref[0] = float(actions[k, h, 2])
                obs, _, term, trunc, _ = main_env.step(actions[k, h, :2])
                traj[h, 0] = obs[0]           # roll [rad]
                traj[h, 1] = obs[1]           # pitch [rad]
                traj[h, 2] = obs[2] / 3.6     # Va [km/h → m/s]
                if term or trunc:
                    break
            buf = traj.tobytes()
            sent = 0
            while sent < len(buf):
                sent += os.write(w_fd, buf[sent:])
            os.close(w_fd)
            os._exit(0)
        os.close(w_fd)                                 # ── parent ──
        pids_and_pipes.append((pid, r_fd))

    trajectories = np.full((n, horizon, 3), np.nan)
    for k, (pid, r_fd) in enumerate(pids_and_pipes):
        os.waitpid(pid, 0)
        data = b''
        while len(data) < bytes_per_traj:
            data += os.read(r_fd, bytes_per_traj - len(data))
        os.close(r_fd)
        trajectories[k] = np.frombuffer(data, dtype=np.float64).reshape(horizon, 3)
    return trajectories


# ── MPPI loop ──────────────────────────────────────────────────────────────────

def run_mppi_oracle(target_roll: float, target_pitch: float, max_steps: int,
                    mppi_cfg: dict, seed: int,
                    render_mode: str = 'none',
                    log_every: int = 100) -> dict:
    """Run the MPPI oracle controller.

    Args:
        target_roll/pitch : attitude setpoints [°]
        max_steps         : total simulation steps
        mppi_cfg          : samples, horizon, temperature, noise_std, iters
        seed              : numpy RNG seed
        render_mode       : JSBSim render mode
        log_every         : print a progress row every N steps (0 = silent)

    Returns a dict with roll/pitch/va/p/q/r histories, actions, step_times,
    steps, terminated.
    """
    with suppress_output():
        main_env = make_env(render_mode=render_mode,
                            telemetry_file='telemetry/mppi_oracle.csv')

    thr_ref, restore_patch = _throttle_patch(main_env)
    controller = MPPIController(
        horizon=mppi_cfg['horizon'], action_dim=3,
        num_samples=mppi_cfg['samples'],
        temperature=mppi_cfg['temperature'],
        noise_std=mppi_cfg['noise_std'],
        min_std=mppi_cfg.get('min_std', 0.05),
        num_elites=mppi_cfg.get('num_elites', 64),
        momentum=mppi_cfg.get('momentum', 0.1),
    )
    np.random.seed(seed)
    controller.reset()

    with suppress_output():
        obs, _ = main_env.reset(options={"fgear_target_roll":  target_roll,
                                          "fgear_target_pitch": target_pitch})

    roll_rad  = np.deg2rad(target_roll)
    pitch_rad = np.deg2rad(target_pitch)
    n_iters   = mppi_cfg.get('iters') or 1
    main_env.unwrapped.set_target_state(np.array([roll_rad, pitch_rad]))

    roll_h, pitch_h, va_h, p_h, q_h, r_h = [], [], [], [], [], []
    act_h, times = [], []
    terminated = False

    print(f"\n{'─'*68}")
    print(f"  Oracle MPPI | roll={target_roll:+.1f}° pitch={target_pitch:+.1f}° "
          f"Va={TARGET_VA_KPH:.0f} kph")
    print(f"  N={mppi_cfg['samples']} H={mppi_cfg['horizon']} "
          f"λ={mppi_cfg['temperature']} σ={mppi_cfg['noise_std']} seed={seed}")
    print(f"{'─'*68}")
    if log_every > 0:
        print(f"  {'step':>5}  {'roll(°)':>8}  {'err_r(°)':>9}  "
              f"{'pitch(°)':>9}  {'err_p(°)':>9}  {'Va(kph)':>8}  {'ms':>6}")
        print(f"  {'─'*63}")

    with tqdm(total=max_steps, desc='MPPI oracle', unit='step', leave=True) as pbar:
        for step in range(max_steps):
            t0   = time.time()
            best = np.zeros(3)

            for i in range(n_iters):
                sampled      = controller.sample_actions()
                trajectories = _rollout_fork(main_env, thr_ref, sampled)
                costs        = compute_costs(trajectories, target_roll, target_pitch)
                best         = controller.update(costs, sampled, shift=(i == n_iters - 1))

            thr_ref[0] = float(best[2])
            obs, _, term, trunc, _ = main_env.step(best[:2])
            dt_ms = (time.time() - t0) * 1000

            r_deg, p_deg = np.rad2deg(obs[0]), np.rad2deg(obs[1])
            roll_h.append(r_deg);  pitch_h.append(p_deg);  va_h.append(obs[2])
            p_h.append(obs[3]);    q_h.append(obs[4]);      r_h.append(obs[5])
            act_h.append(best.copy()); times.append(time.time() - t0)

            pbar.set_postfix(
                roll=f'{r_deg:+.1f}°',
                err_r=f'{abs(r_deg-target_roll):.1f}°',
                pitch=f'{p_deg:+.1f}°',
                err_p=f'{abs(p_deg-target_pitch):.1f}°',
                Va=f'{obs[2]:.1f}',
                ms=f'{dt_ms:.0f}',
            )
            pbar.update(1)

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

    return dict(label='JSBSim_fork',
                roll=roll_h, pitch=pitch_h, va=va_h,
                p=p_h, q=q_h, r=r_h,
                actions=np.array(act_h), step_times=times,
                steps=len(roll_h), terminated=terminated)


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(n_warmup: int = 50, n_test: int = 20,
                 target_roll: float = 55.0, target_pitch: float = 28.0) -> bool:
    """Verify that os.fork() produces a bit-perfect clone at a mid-run state.

    Warms up one env for n_warmup steps, then forks: child and parent each run
    the same n_test actions. Fork diff vs ground truth must be exactly zero.
    """
    t0 = time.time()
    print(f"\n{'═'*65}")
    print(f"  SANITY CHECK — os.fork() clone fidelity at mid-run state")
    print(f"  Warm-up: {n_warmup} steps    Test seq: {n_test} actions    seed=0")
    print(f"{'═'*65}")

    target_r = np.deg2rad(target_roll)
    target_p = np.deg2rad(target_pitch)

    with suppress_output():
        env_main = make_env()

    thr_main, _ = _throttle_patch(env_main)

    with suppress_output():
        env_main.reset(options={"fgear_target_roll":  target_roll,
                                "fgear_target_pitch": target_pitch})
    env_main.unwrapped.set_target_state(np.array([target_r, target_p]))

    print(f"\n  [1/2] Warming up {n_warmup} steps ...", end='', flush=True)
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

    rng  = np.random.default_rng(seed=0)
    acts = np.column_stack([rng.uniform(-0.8, 0.8, n_test),
                             rng.uniform(-0.6, 0.6, n_test),
                             rng.uniform( 0.2, 0.7, n_test)])

    print(f"\n  [2/2] os.fork() rollout ...", end='', flush=True)
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
    env_main.close()

    fork_d = [np.abs(np.array(parent_obs[k]) - np.array(fork_obs[k])) for k in range(n_test)]

    OBS = ['roll(rad)', 'pitch(rad)', 'Va(kph)', 'p(rad/s)', 'r(rad/s)']
    IDX = [0, 1, 2, 3, 5]
    W   = 10
    hdr = "  ".join(f"{n:^{W}}" for n in OBS)
    sep = "─" * (8 + len(OBS) * (W + 2))

    print(f"\n  ── os.fork() | diff vs ground truth (expect all zeros) ──")
    print(f"  {'step':>4}  {hdr}")
    print(f"  {sep}")
    for k in range(n_test):
        row = "  ".join(f"{fork_d[k][i]:>{W}.2e}" for i in IDX)
        print(f"  {k+1:>4}  {row}")
    fork_max = max(fork_d[k][i] for k in range(n_test) for i in IDX)
    print(f"  MAX over all channels: {fork_max:.2e}")

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
    parser = argparse.ArgumentParser(description='MPPI Oracle — JSBSim as world model')
    parser.add_argument('--target-roll',      type=float, default=20.0)
    parser.add_argument('--target-pitch',     type=float, default=10.0)
    parser.add_argument('--steps',            type=int,   default=2000)
    parser.add_argument('--mppi-samples',     type=int,   default=512)
    parser.add_argument('--mppi-horizon',     type=int,   default=40)
    parser.add_argument('--mppi-temperature', type=float, default=0.5)
    parser.add_argument('--mppi-noise-std',   type=float, default=0.4)
    parser.add_argument('--min-std',          type=float, default=0.05,
                        help='σ floor — prevents over-exploitation (eq. 5 of TD-MPC)')
    parser.add_argument('--mppi-iters',       type=int,   default=6,
                        help='MPPI refinement passes per step (default: 6)')
    parser.add_argument('--num-elites',       type=int,   default=64,
                        help='Top-k trajectories used for μ/σ update (TD-MPC: 64 of 512)')
    parser.add_argument('--momentum',         type=float, default=0.1,
                        help='Mean momentum across iterations (TD-MPC: 0.1)')
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--render-mode',      type=str,   default='none',
                        choices=['none', 'plot_anim', 'plot_end',
                                 'ext_log', 'fgear', 'fgear_plot'])
    parser.add_argument('--log-every',        type=int,   default=1)
    parser.add_argument('--save-dir',         type=str,   default=str(SAVE_DIR),
                        help='Directory to save plot and metrics CSV')
    parser.add_argument('--sanity',           action='store_true',
                        help='Run fork clone sanity check and exit')
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
        min_std=args.min_std, num_elites=args.num_elites,
        momentum=args.momentum, iters=args.mppi_iters,
    )

    result = run_mppi_oracle(
        args.target_roll, args.target_pitch, args.steps,
        mppi_cfg, args.seed,
        render_mode=args.render_mode,
        log_every=args.log_every,
    )

    tag = f"r{args.target_roll:.0f}_p{args.target_pitch:.0f}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_model_result(result, args.target_roll, args.target_pitch,
                      save_dir / f"{_safe_label(result['label'])}_{tag}.png",
                      target_va=TARGET_VA_KPH)
    save_metrics([result], args.target_roll, args.target_pitch,
                 save_dir / f"metrics_{tag}_{_safe_label(result['label'])}.csv")


if __name__ == '__main__':
    main()
