#!/usr/bin/env python3
import sys, time, argparse
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import fw_jsbgym  # noqa: F401

from fw_flightcontrol.physics.mppi import (
    MPPIController, load_config_and_model, initialize_environment,
    rollout_trajectories, compute_costs,
)
from fw_flightcontrol.physics.utils import (
    _throttle_patch, _safe_label,
    plot_model_result, save_metrics, compute_convergence_stats,
)


DT            = 0.01
TARGET_VA_KPH = 60.0
SAVE_DIR      = Path(__file__).parent.parent.parent / 'data' / 'test_hybrid_mppi_topk_highernoise'


# ── Control loop ───────────────────────────────────────────────────────────────

def run_mppi_hybrid(
    env,
    hybrid_model,
    model_config,
    norm_scale,
    norm_offset,
    norm_type,
    target_roll:  float,
    target_pitch: float,
    max_steps:    int,
    mppi_cfg:     dict,
    seed:         int,
    device:       torch.device,
    log_every:    int = 1,
) -> dict:
    """Run one MPPI episode using the hybrid model for rollouts.

    Prints per-step state and a summary at the end.
    Returns a result dict in the same format as run_mppi_oracle.

    Args:
        env           : initialized JSBSim gymnasium environment
        hybrid_model  : loaded HybridDynamicsModel (eval mode)
        model_config  : training config dict
        norm_scale / norm_offset / norm_type : normalization from checkpoint
        target_roll/pitch : attitude setpoints [°]
        max_steps     : maximum simulation steps
        mppi_cfg      : samples, horizon, temperature, noise_std, residual_clamp
        seed          : numpy RNG seed
        device        : torch device
        log_every     : print a row every N steps (0 = silent)
    """
    thr_ref, restore = _throttle_patch(env)

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

    obs, _ = env.reset(options={"fgear_target_roll":  target_roll,
                                 "fgear_target_pitch": target_pitch})
    env.unwrapped.set_target_state(
        np.array([np.deg2rad(target_roll), np.deg2rad(target_pitch)])
    )

    n_iters = mppi_cfg.get('iters') or 1
    roll_h, pitch_h, va_h, p_h, q_h, r_h = [], [], [], [], [], []
    act_h, times = [], []
    terminated = False

    print(f"\n{'─'*68}")
    print(f"  Hybrid MPPI | roll={target_roll:+.1f}° pitch={target_pitch:+.1f}°")
    print(f"  N={mppi_cfg['samples']} H={mppi_cfg['horizon']} "
          f"λ={mppi_cfg['temperature']} σ={mppi_cfg['noise_std']} "
          f"iters={n_iters} seed={seed}")
    print(f"{'─'*68}")
    if log_every > 0:
        print(f"  {'step':>5}  {'roll(°)':>8}  {'err_r(°)':>9}  "
              f"{'pitch(°)':>9}  {'err_p(°)':>9}  {'Va(kph)':>8}  {'ms':>6}")
        print(f"  {'─'*63}")

    with tqdm(total=max_steps, desc='MPPI hybrid', unit='step', leave=True) as pbar:
        for step in range(max_steps):
            t0 = time.time()

            for i in range(n_iters):
                sampled = controller.sample_actions()
                trajs   = rollout_trajectories(
                    obs, sampled, hybrid_model, model_config,
                    norm_scale, norm_offset, norm_type, device,
                    residual_clamp=mppi_cfg.get('residual_clamp'),
                )
                costs = compute_costs(trajs, target_roll, target_pitch)
                best  = controller.update(costs, sampled, shift=(i == n_iters - 1))

            thr_ref[0] = float(best[2])
            obs, _, term, trunc, _ = env.step(best[:2])
            dt_ms = (time.time() - t0) * 1000

            r_deg, p_deg = np.rad2deg(obs[0]), np.rad2deg(obs[1])
            roll_h.append(r_deg); pitch_h.append(p_deg); va_h.append(obs[2])
            p_h.append(obs[3]);   q_h.append(obs[4]);    r_h.append(obs[5])
            act_h.append(best.copy()); times.append(time.time() - t0)

            pbar.set_postfix(
                roll=f'{r_deg:+.1f}°',
                err_r=f'{abs(r_deg - target_roll):.1f}°',
                pitch=f'{p_deg:+.1f}°',
                err_p=f'{abs(p_deg - target_pitch):.1f}°',
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

    restore()
    env.close()

    if times:
        ra = np.array(roll_h); pa = np.array(pitch_h)
        rc = compute_convergence_stats(roll_h,  target_roll)
        pc = compute_convergence_stats(pitch_h, target_pitch)
        n  = len(times)
        r_conv = f"{rc['convergence_step']*DT:.1f} s" if rc['convergence_step'] else 'never'
        p_conv = f"{pc['convergence_step']*DT:.1f} s" if pc['convergence_step'] else 'never'
        print(f"\n  {'─'*63}")
        print(f"  Summary  ({n} steps{', terminated early' if terminated else ''})")
        print(f"  Roll  : mean err={np.mean(np.abs(ra-target_roll)):.2f}°  "
              f"converge={r_conv}  "
              f"ss={rc['steady_mean_error']:.2f}°±{rc['steady_std']:.2f}°")
        print(f"  Pitch : mean err={np.mean(np.abs(pa-target_pitch)):.2f}°  "
              f"converge={p_conv}  "
              f"ss={pc['steady_mean_error']:.2f}°±{pc['steady_std']:.2f}°")
        print(f"  Step  : mean={np.mean(times)*1000:.0f} ms  "
              f"min={np.min(times)*1000:.0f} ms  max={np.max(times)*1000:.0f} ms")
        print(f"  {'─'*63}\n")

    return dict(label='hybrid_model',
                roll=roll_h, pitch=pitch_h, va=va_h,
                p=p_h, q=q_h, r=r_h,
                actions=np.array(act_h) if act_h else np.empty((0, 3)),
                step_times=times,
                steps=len(roll_h), terminated=terminated)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MPPI with hybrid dynamics model')
    _dir = Path(__file__).parent.parent
    parser.add_argument('--checkpoint',        type=str,
                        default=str(_dir / 'models/data_norm_model.pt'),
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--config',            type=str,
                        default=str(_dir / 'training_params.yaml'),
                        help='Path to training_params.yaml')
    parser.add_argument('--target-roll',       type=float, default=20.0)
    parser.add_argument('--target-pitch',      type=float, default=10.0)
    parser.add_argument('--steps',             type=int,   default=2000)
    parser.add_argument('--mppi-samples',      type=int,   default=512)
    parser.add_argument('--mppi-horizon',      type=int,   default=20)
    parser.add_argument('--mppi-temperature',  type=float, default=0.5)
    parser.add_argument('--mppi-noise-std',    type=float, default=0.4)
    parser.add_argument('--min-std',           type=float, default=0.05,
                        help='sig floor — prevents over-exploitation (eq. 5 of TD-MPC)')
    parser.add_argument('--mppi-iters',        type=int,   default=1,
                        help='MPPI refinement passes per step (default: 1)')
    parser.add_argument('--num-elites',        type=int,   default=64,
                        help='Top-k trajectories used for mu/sig update (TD-MPC: 64 of 512)')
    parser.add_argument('--momentum',          type=float, default=0.1,
                        help='Mean momentum across iterations (TD-MPC: 0.1)')
    parser.add_argument('--residual-clamp',    type=float, default=None,
                        help='Clamp residual output to [-x, x] to prevent OOD explosion')
    parser.add_argument('--seed',              type=int,   default=42)
    parser.add_argument('--device',            type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-every',         type=int,   default=1)
    args = parser.parse_args()

    device = torch.device(args.device)

    hybrid_model, model_config, norm_scale, norm_offset, norm_type = load_config_and_model(
        args.checkpoint, args.config, device
    )

    try:
        from hydra import compose, initialize_config_dir
        config_dir = str(Path(__file__).parent.parent.parent / 'config')
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='default')
    except Exception:
        cfg = OmegaConf.create({'env': {}})

    env = initialize_environment(cfg)
    if env is None:
        print("Failed to initialize environment. Exiting.")
        return

    mppi_cfg = dict(
        samples=args.mppi_samples,
        horizon=args.mppi_horizon,
        temperature=args.mppi_temperature,
        noise_std=args.mppi_noise_std,
        min_std=args.min_std,
        num_elites=args.num_elites,
        momentum=args.momentum,
        iters=args.mppi_iters,
        residual_clamp=args.residual_clamp,
    )

    result = run_mppi_hybrid(
        env, hybrid_model, model_config, norm_scale, norm_offset, norm_type,
        target_roll=args.target_roll,
        target_pitch=args.target_pitch,
        max_steps=args.steps,
        mppi_cfg=mppi_cfg,
        seed=args.seed,
        device=device,
        log_every=args.log_every,
    )

    tag = f"r{args.target_roll:.0f}_p{args.target_pitch:.0f}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_result(result, args.target_roll, args.target_pitch,
                      SAVE_DIR / f"{_safe_label(result['label'])}_{tag}.png",
                      target_va=TARGET_VA_KPH)
    save_metrics([result], args.target_roll, args.target_pitch,
                 SAVE_DIR / f"metrics_{tag}_{_safe_label(result['label'])}.csv")


if __name__ == '__main__':
    main()

'''
Notes:

for hybrid model: iterations seems to hurt the model, 
higher noise (0.5) allows to reach the target but we get oscillations and model cant
seem to converge
with low noise it converges but always to a value slightly lower (2.5° lower than the target)
low noise also gives less action oscillation

for temperature 0.5 seems best

concerning top k candidates (testing rn need to update)

concenrining horizon, 40 works best, 20 degrades only slightly

'''