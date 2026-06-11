#!/usr/bin/env python3
"""
Evaluation script for hybrid physics-augmented world model.

Loads a trained residual network checkpoint from a .pt path and evaluates it
over the full dataset (one epoch), reporting per-batch and aggregate statistics
directly to the terminal. No TensorBoard logging.

Usage:
    python test_physics_model.py --checkpoint path/to/model.pt
    python test_physics_model.py --checkpoint path/to/epoch_100.pt --split val
    python test_physics_model.py --checkpoint path/to/model.pt --with-prior false
"""

import torch
import sys
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import HybridDynamicsODE
from fw_flightcontrol.physics.utils import (
    load_config, load_trajectory_data, STATE_NAMES,
    normalize_state_torch, denormalize_state_torch,
    clean_state_dict_for_compilation,
)


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_model(
    config: Dict,
    checkpoint_path: str,
    device: torch.device,
    with_prior: bool = True,
    with_residual: bool = True,
) -> HybridDynamicsModel:
    """Build the hybrid model and load residual network weights from checkpoint.

    Norm parameters from the checkpoint are attached to the returned model as
    hybrid_model.norm_scale / hybrid_model.norm_offset.
    """
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)

    physics_prior = PhysicsPrior()
    print("  ✓ Physics prior loaded (frozen)")

    print(f"\n  Loading weights from: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw, dict) and 'residual_state' in raw:
        residual_state = clean_state_dict_for_compilation(raw['residual_state'])
        saved_epoch    = raw.get('epoch', '?')
        saved_lambda   = raw.get('lambda', '?')
        norm_scale  = torch.tensor(raw['norm_scale'],  dtype=torch.float32, device=device)
        norm_offset = torch.tensor(raw['norm_offset'], dtype=torch.float32, device=device)
        print(f"  ✓ Loaded training checkpoint (epoch={saved_epoch}, λ={saved_lambda})")
        print("  ✓ Normalization parameters loaded from checkpoint")
    else:
        residual_state = clean_state_dict_for_compilation(raw)
        norm_scale = norm_offset = None
        print("  ✓ Loaded bare state dict — no norm parameters in checkpoint")

    net_config = config['network']
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=net_config['hidden_dims'],
        activation=net_config['activation'],
        use_batch_norm=net_config['use_batch_norm'],
    )
    residual_network.load_state_dict(residual_state)
    num_params = sum(p.numel() for p in residual_network.parameters())
    print(f"  ✓ Residual network: {num_params:,} parameters")

    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=with_prior,
        with_residual=with_residual,
        integration_method=config.get('integration', {}).get('method', 'rk4'),
    ).to(device).eval()

    hybrid_model.norm_scale  = norm_scale
    hybrid_model.norm_offset = norm_offset

    print(f"  ✓ Hybrid model ready on {device} "
          f"(with_prior={with_prior}, with_residual={with_residual})")
    return hybrid_model


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    hybrid_model: HybridDynamicsModel,
    loader,
    config: Dict,
    device: torch.device,
    split_name: str = "eval",
) -> Dict:
    """Evaluate model on a dataset split, computing compounded multi-step errors."""
    horizon    = config['training']['horizon']
    ode_method = config['integration']['method']
    ode_rtol   = config['integration']['rtol']
    ode_atol   = config['integration']['atol']
    dt         = config['integration']['dt']
    lambda_val = config['aphynity'].get('lambda_init', 1.0)
    state_dim  = config['network']['state_dim']

    norm_scale  = hybrid_model.norm_scale
    norm_offset = hybrid_model.norm_offset

    print(f"\n{'='*60}")
    print(f"EVALUATION  [{split_name.upper()}]  —  {len(loader)} batches")
    print(f"{'='*60}")
    print(f"{'Batch':>6}  {'L_traj':>10}  {'L_reg':>10}  {'L_total':>10}  {'RunAvg_traj':>12}")
    print("-" * 56)

    all_traj_losses  = []
    all_reg_losses   = []
    all_total_losses = []

    per_state_abs_err_sum = np.zeros(state_dim)
    per_state_sq_err_sum  = np.zeros(state_dim)
    per_state_abs_gt_sum  = np.zeros(state_dim)
    per_state_count       = 0

    horizon_steps    = [h for h in [1, 3, 5, 10] if h <= horizon]
    per_horizon_err  = {h: [] for h in horizon_steps}
    per_horizon_gt   = {h: [] for h in horizon_steps}

    for batch_idx, batch_data in enumerate(loader):
        batch_initial = batch_data['initial_states'].to(device)
        batch_actions = batch_data['actions'].to(device)
        batch_states  = batch_data['states'].to(device)

        predicted_states = []
        residual_norms   = []
        current_state    = batch_initial

        ode_module = HybridDynamicsODE(hybrid_model, device).to(device)
        t_eval = torch.tensor([0.0, dt], dtype=current_state.dtype, device=device)

        for step in range(horizon):
            action = batch_actions[:, step, :]

            current_state_raw = denormalize_state_torch(current_state, norm_scale, norm_offset)
            ode_module.set_action(action)
            ode_module.arm_capture()
            solution = odeint(ode_module, current_state_raw, t_eval,
                              method=ode_method, rtol=ode_rtol, atol=ode_atol)
            next_state_raw = solution[-1].clamp(-1000.0, 1000.0)

            if hybrid_model.with_residual and ode_module.captured_residual_norm is not None:
                residual_norms.append(ode_module.captured_residual_norm)

            next_state = normalize_state_torch(next_state_raw, norm_scale, norm_offset)
            predicted_states.append(next_state)
            current_state = next_state

        predicted_trajectory = torch.stack(predicted_states, dim=1)  # (B, H, state_dim)

        # Loss in normalized space (matches training)
        traj_loss  = torch.norm(predicted_trajectory - batch_states, p=2, dim=2).mean().item()
        reg_loss   = torch.stack(residual_norms).mean().item() if residual_norms else 0.0
        total_loss = reg_loss + lambda_val * traj_loss

        all_traj_losses.append(traj_loss)
        all_reg_losses.append(reg_loss)
        all_total_losses.append(total_loss)

        # Per-state errors in raw space for human-readable accuracy table
        pred_raw = denormalize_state_torch(predicted_trajectory, norm_scale, norm_offset)
        gt_raw   = denormalize_state_torch(batch_states,         norm_scale, norm_offset)
        err_np   = (pred_raw - gt_raw).cpu().numpy()
        gt_np    = gt_raw.cpu().numpy()
        err_flat = err_np.reshape(-1, state_dim)
        gt_flat  = gt_np.reshape(-1, state_dim)

        per_state_abs_err_sum += np.abs(err_flat).sum(axis=0)
        per_state_sq_err_sum  += (err_flat ** 2).sum(axis=0)
        per_state_abs_gt_sum  += np.abs(gt_flat).sum(axis=0)
        per_state_count       += err_flat.shape[0]

        for h in horizon_steps:
            if h - 1 < err_np.shape[1]:
                per_horizon_err[h].append(err_np[:, h-1, :])
                per_horizon_gt[h].append(gt_np[:, h-1, :])

        running_avg_traj = np.mean(all_traj_losses)
        print(f"{batch_idx+1:>6}  {traj_loss:>10.4f}  {reg_loss:>10.4f}  {total_loss:>10.4f}  {running_avg_traj:>12.4f}")

    print("-" * 56)

    # -------------------------------------------------------------------------
    # Global summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY  [{split_name.upper()}]")
    print(f"{'='*60}")
    print(f"  Batches evaluated   : {len(all_traj_losses)}")
    print(f"  Avg trajectory loss : {np.mean(all_traj_losses):.6f}  (±{np.std(all_traj_losses):.6f})")
    print(f"  Avg regularization  : {np.mean(all_reg_losses):.6f}  (±{np.std(all_reg_losses):.6f})")
    print(f"  Avg total loss      : {np.mean(all_total_losses):.6f}  (±{np.std(all_total_losses):.6f})")
    print(f"  Min trajectory loss : {np.min(all_traj_losses):.6f}")
    print(f"  Max trajectory loss : {np.max(all_traj_losses):.6f}")

    # -------------------------------------------------------------------------
    # Per-state accuracy table (raw space, with per-horizon relative error)
    # -------------------------------------------------------------------------
    per_state_mae      = per_state_abs_err_sum / per_state_count
    per_state_rmse     = np.sqrt(per_state_sq_err_sum / per_state_count)
    per_state_mean_mag = per_state_abs_gt_sum / per_state_count
    per_state_rel_err  = np.where(
        per_state_mean_mag > 1e-6,
        100.0 * per_state_mae / per_state_mean_mag,
        np.nan,
    )

    per_horizon_rel_err = {}
    for h in horizon_steps:
        if per_horizon_err[h]:
            err_h      = np.concatenate(per_horizon_err[h], axis=0)
            gt_h       = np.concatenate(per_horizon_gt[h],  axis=0)
            mae_h      = np.abs(err_h).mean(axis=0)
            mean_mag_h = np.abs(gt_h).mean(axis=0)
            per_horizon_rel_err[h] = np.where(mean_mag_h > 1e-6, 100.0 * mae_h / mean_mag_h, np.nan)
        else:
            per_horizon_rel_err[h] = np.full(state_dim, np.nan)

    print(f"\n{'='*72}")
    print(f"PER-STATE ACCURACY  [{split_name.upper()}]")
    print(f"  COMPOUNDED ERRORS: predictions use previous predictions (matches training loss)")
    print(f"  (averaged over all {per_state_count} predictions = batches × horizon)")
    print(f"  Relative error = MAE / mean(|ground truth|)   [raw physical units]")
    print(f"{'='*72}")

    header = f"  {'State':<22}  {'MAE':>10}  {'RMSE':>10}  {'Mean |gt|':>10}  {'Rel err %':>10}"
    for h in horizon_steps:
        header += f"  [h={h}]%"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(STATE_NAMES):
        rel_str = f"{per_state_rel_err[i]:>9.2f}%" if not np.isnan(per_state_rel_err[i]) else "       N/A"
        row = f"  {name:<22}  {per_state_mae[i]:>10.6f}  {per_state_rmse[i]:>10.6f}  {per_state_mean_mag[i]:>10.4f}  {rel_str}"
        for h in horizon_steps:
            rel_h     = per_horizon_rel_err[h][i]
            rel_h_str = f"{rel_h:>8.2f}%" if not np.isnan(rel_h) else "    N/A"
            row += f"  {rel_h_str}"
        print(row)

    print(f"{'='*72}\n")

    return {
        'mean_traj_loss':     np.mean(all_traj_losses),
        'std_traj_loss':      np.std(all_traj_losses),
        'mean_reg_loss':      np.mean(all_reg_losses),
        'mean_total_loss':    np.mean(all_total_losses),
        'per_state_mae':      per_state_mae,
        'per_state_rmse':     per_state_rmse,
        'per_state_mean_mag': per_state_mean_mag,
        'per_state_rel_err':  per_state_rel_err,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained hybrid physics-augmented world model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint (training checkpoint or final_model.pt)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to trajectory CSV file (default: physics/data/updated_trajectory_data_progressive_noatmo_2.0.csv)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which data split to evaluate on (default: val)')
    parser.add_argument('--with-prior',    type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='Include physics prior in model (default: True)')
    parser.add_argument('--with-residual', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='Include learned residual in model (default: True)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("HYBRID PHYSICS-AUGMENTED MODEL — EVALUATION")
    print("="*60)
    print(f"  Checkpoint       : {args.checkpoint}")
    print(f"  Split            : {args.split}")
    print(f"  Physics prior    : {args.with_prior}")
    print(f"  Residual network : {args.with_residual}")

    config_path = Path(__file__).parent.parent / 'training_params.yaml'
    config = load_config(str(config_path))
    print(f"  Config           : {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device           : {device}")

    csv_path = args.data or str(
        Path(__file__).parent.parent / "data" / "updated_trajectory_data_progressive_noatmo_2.0.csv"
    )
    print(f"  Data             : {csv_path}")

    train_loader, val_loader, test_loader, _, _ = load_trajectory_data(csv_path, config)

    hybrid_model = initialize_model(
        config, args.checkpoint, device,
        with_prior=args.with_prior,
        with_residual=args.with_residual,
    )

    if hybrid_model.norm_scale is not None:
        print(f"\n  norm_scale  (std):    {hybrid_model.norm_scale.tolist()}")
        print(f"  norm_offset (mean):   {hybrid_model.norm_offset.tolist()}")

    loader_map    = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    target_loader = loader_map[args.split]

    evaluate(hybrid_model, target_loader, config, device, split_name=args.split)


if __name__ == '__main__':
    main()
