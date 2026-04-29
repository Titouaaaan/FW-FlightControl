#!/usr/bin/env python3
"""
Evaluation script for hybrid physics-augmented world model.

Loads a trained residual network checkpoint from a .pt path and evaluates it
over the full dataset (one epoch), reporting per-batch and aggregate statistics
directly to the terminal. No TensorBoard logging.

Usage:
    python evaluate_model.py --checkpoint path/to/model.pt
    python evaluate_model.py --checkpoint path/to/epoch_100.pt --split test
"""

import torch
import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import pandas as pd

# Add project to path
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import HybridDynamicsODE


# ============================================================================
# NORMALIZATION UTILITIES (copied from learn_physics_model.py)
# ============================================================================
def extract_bounds_from_config(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract state bounds from config and return as min/max arrays.
    
    Returns:
        (min_bounds, max_bounds): numpy arrays of shape (state_dim,)
        Order: [roll, pitch, airspeed_mps, p, q, r, alpha, beta]
    """
    bounds = config['state_bounds']
    
    min_bounds = np.array([
        bounds['roll_min'],
        bounds['pitch_min'],
        bounds['airspeed_mps_min'],
        bounds['p_min'],
        bounds['q_min'],
        bounds['r_min'],
        bounds['alpha_min'],
        bounds['beta_min'],
    ], dtype=np.float32)
    
    max_bounds = np.array([
        bounds['roll_max'],
        bounds['pitch_max'],
        bounds['airspeed_mps_max'],
        bounds['p_max'],
        bounds['q_max'],
        bounds['r_max'],
        bounds['alpha_max'],
        bounds['beta_max'],
    ], dtype=np.float32)
    
    return min_bounds, max_bounds


def compute_denorm_factors(min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
    """
    Compute denormalization factors for derivatives.
    
    For a state normalized to [-1, 1] as: s_norm = 2*(s_raw - min)/(max-min) - 1
    The derivative scaling is: ds_norm/dt = 2/(max-min) * ds_raw/dt
    To denormalize: ds_raw/dt = (max-min)/2 * ds_norm/dt
    
    Returns:
        denorm_factors: array of shape (state_dim,) with (max-min)/2 for each state
    """
    return (max_bounds - min_bounds) / 2.0


def compute_actual_scales(df: pd.DataFrame) -> np.ndarray:
    """
    Compute actual per-state scales from training data (mean absolute value).
    
    These reflect what's actually in the data, not config bounds.
    Important for per-state loss scaling: states with small actual variation
    get amplified gradients, states with large variation get dampened gradients.
    
    Args:
        df: DataFrame with state columns
    
    Returns:
        actual_scales: array of shape (state_dim,) with mean |value| for each state
    """
    state_indices = [0, 1, 2, 3, 4, 5, 8, 9]
    state_cols = [f's_t_{i}' for i in state_indices]
    
    actual_scales = []
    for col in state_cols:
        values = df[col].values.copy()
        # Convert airspeed from km/h to m/s
        if col == 's_t_2':
            values = values / 3.6
        mean_abs = np.mean(np.abs(values))
        actual_scales.append(mean_abs)
    
    return np.array(actual_scales, dtype=np.float32)


def normalize_state(state: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
    """
    Normalize state from raw values to [-1, 1] range.
    
    Formula: normalized = 2 * (value - min) / (max - min) - 1
    
    Args:
        state: raw state values, shape (..., state_dim)
        min_bounds: minimum bounds, shape (state_dim,)
        max_bounds: maximum bounds, shape (state_dim,)
    
    Returns:
        normalized state in [-1, 1] range
    """
    return 2.0 * (state - min_bounds) / (max_bounds - min_bounds) - 1.0


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trajectory_data(csv_path: str, config: Dict) -> Tuple:
    """
    Load trajectory data from CSV and prepare train/val/test splits.
    Identical logic to the training script.
    """
    import pandas as pd

    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transitions from {df['trajectory_id'].nunique()} trajectories")

    horizon   = config['training']['horizon']
    state_dim = config['network']['state_dim']
    action_dim = config['network']['action_dim']
    normalize  = config['data']['normalize']
    batch_size = config['training']['batch_size']

    state_indices   = [0, 1, 2, 3, 4, 5, 8, 9]
    state_cols      = [f's_t_{i}' for i in state_indices]
    action_cols     = [f'a_t_{i}' for i in range(action_dim)]
    next_state_cols = [f's_t+1_{i}' for i in state_indices]

    # Extract bounds for normalization (from config)
    min_bounds, max_bounds = extract_bounds_from_config(config)
    denorm_factors = compute_denorm_factors(min_bounds, max_bounds)
    
    print(f"\nState bounds loaded from config:")
    print(f"  Min: {min_bounds}")
    print(f"  Max: {max_bounds}")
    print(f"  Denorm factors: {denorm_factors}")

    trajectory_sequences_by_traj = {}

    for traj_id, group in df.groupby('trajectory_id'):
        group = group.sort_values('step_id').reset_index(drop=True)

        states      = group[state_cols].values.copy()
        actions     = group[action_cols].values
        next_states = group[next_state_cols].values.copy()

        states[:, 2]      = states[:, 2] / 3.6
        next_states[:, 2] = next_states[:, 2] / 3.6

        num_steps    = len(states)
        traj_sequences = []

        for start_idx in range(num_steps - horizon):
            seq_states      = states[start_idx:start_idx + horizon]
            seq_actions     = actions[start_idx:start_idx + horizon]
            seq_next_states = next_states[start_idx:start_idx + horizon]

            # Normalize if configured (bounds-based min-max to [-1, 1])
            if normalize:
                seq_states      = normalize_state(seq_states, min_bounds, max_bounds)
                seq_next_states = normalize_state(seq_next_states, min_bounds, max_bounds)

            traj_sequences.append({
                'initial_state': seq_states[0].copy(),
                'actions':       seq_actions.copy(),
                'states':        seq_next_states.copy(),
            })

        trajectory_sequences_by_traj[traj_id] = traj_sequences

    trajectories = sorted(list(df['trajectory_id'].unique()))
    random_seed = config['data'].get('random_seed', 42)
    np.random.seed(random_seed)
    np.random.shuffle(trajectories)

    n_traj_train = int(len(trajectories) * config['data']['train_fraction'])
    n_traj_val   = int(len(trajectories) * config['data']['val_fraction'])

    train_traj_ids = set(trajectories[:n_traj_train])
    val_traj_ids   = set(trajectories[n_traj_train:n_traj_train + n_traj_val])
    test_traj_ids  = set(trajectories[n_traj_train + n_traj_val:])

    def build_seqs(traj_ids):
        return [seq
                for traj_id, seq_list in trajectory_sequences_by_traj.items()
                if traj_id in traj_ids
                for seq in seq_list]

    train_seqs = build_seqs(train_traj_ids)
    val_seqs   = build_seqs(val_traj_ids)
    test_seqs  = build_seqs(test_traj_ids)

    print(f"\nTrain/Val/Test split (trajectory-level):")
    print(f"  Trajectories: {len(train_traj_ids)} train | {len(val_traj_ids)} val | {len(test_traj_ids)} test")
    print(f"  Sequences:    {len(train_seqs)} train | {len(val_seqs)} val | {len(test_seqs)} test")

    class TrajectoryDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            seq = self.sequences[idx]
            return {
                'initial_states': torch.tensor(seq['initial_state'], dtype=torch.float32),
                'actions':        torch.tensor(seq['actions'],        dtype=torch.float32),
                'states':         torch.tensor(seq['states'],         dtype=torch.float32),
            }

        @staticmethod
        def collate_fn(batch):
            return {
                'initial_states': torch.stack([b['initial_states'] for b in batch]),
                'actions':        torch.stack([b['actions']        for b in batch]),
                'states':         torch.stack([b['states']         for b in batch]),
            }

    def make_loader(seqs, shuffle=False):
        ds = TrajectoryDataset(seqs)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=TrajectoryDataset.collate_fn,
        )

    return make_loader(train_seqs), make_loader(val_seqs), make_loader(test_seqs), denorm_factors, min_bounds


def initialize_models(config: Dict, checkpoint_path: str, device: torch.device,
                      with_prior: bool = True, with_residual: bool = True) -> HybridDynamicsModel:
    """
    Build the hybrid model and load residual network weights from checkpoint.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model weights checkpoint
        device: Device to load model on
        with_prior: Whether to include physics prior (ablation)
        with_residual: Whether to include learned residual (ablation)
    The checkpoint may be a full training checkpoint (dict) or a bare state_dict.
    """
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)

    physics_prior = PhysicsPrior()
    print("  ✓ Physics prior loaded (frozen)")

    net_config = config['network']
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=net_config['hidden_dims'],
        activation=net_config['activation'],
        use_batch_norm=net_config['use_batch_norm'],
    )

    # Load checkpoint — handle both full training checkpoints and bare state dicts
    print(f"\n  Loading weights from: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw, dict) and 'residual_state' in raw:
        # Full training checkpoint saved by train_hybrid_model.py
        residual_network.load_state_dict(raw['residual_state'])
        saved_epoch = raw.get('epoch', '?')
        saved_lambda = raw.get('lambda', '?')
        print(f"  ✓ Loaded from training checkpoint (epoch={saved_epoch}, λ={saved_lambda})")
    else:
        # Bare state dict saved by final_model.pt logic
        residual_network.load_state_dict(raw)
        print("  ✓ Loaded bare state dict (final_model.pt format)")

    num_params = sum(p.numel() for p in residual_network.parameters())
    print(f"  ✓ Residual network: {num_params:,} parameters")

    integration_method = config.get('integration', {}).get('method', 'rk4')
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=with_prior,
        with_residual=with_residual,
        integration_method=integration_method,
    )
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()

    print(f"  ✓ Hybrid model ready on {device}")
    return hybrid_model


@torch.no_grad()
def evaluate(hybrid_model: HybridDynamicsModel, loader, config: Dict, device: torch.device,
             split_name: str = "eval", denorm_factors: torch.Tensor = None, min_bounds: torch.Tensor = None,
             per_state_scales: torch.Tensor = None) -> Dict:
    """Evaluate model on a dataset split, computing compounded errors."""
    from torchdiffeq import odeint

    horizon    = config['training']['horizon']
    ode_method = config['integration']['method']
    ode_rtol   = config['integration']['rtol']
    ode_atol   = config['integration']['atol']
    dt         = config['integration']['dt']
    lambda_val = config['aphynity'].get('lambda_init', 1.0)
    state_dim  = config['network']['state_dim']

    # denorm_factors and min_bounds are already torch tensors on the correct device,
    # passed in from main(). Just alias them for clarity throughout this function.
    # denorm_factors is non-None when normalize=True OR per_state_loss_norm=True.
    # min_bounds is non-None only when normalize=True.
    denorm_factors_torch = denorm_factors
    min_bounds_torch = min_bounds

    # Human-readable names for the 8 state components
    # State indices kept from CSV: [0,1,2,3,4,5,8,9]
    STATE_NAMES = [
        "roll  (s0) [rad]",
        "pitch (s1) [rad]",
        "Va    (s2) [m/s]",
        "p     (s3) [rad/s]",
        "q     (s4) [rad/s]",
        "r     (s5) [rad/s]",
        "AoA   (s8) [rad]",
        "AoS   (s9) [rad]",
    ]

    print(f"\n{'='*60}")
    print(f"EVALUATION  [{split_name.upper()}]  —  {len(loader)} batches")
    print(f"{'='*60}")
    print(f"{'Batch':>6}  {'L_traj':>10}  {'L_reg':>10}  {'L_total':>10}  {'RunAvg_traj':>12}")
    print("-" * 56)

    all_traj_losses  = []
    all_reg_losses   = []
    all_total_losses = []

    # Accumulators for per-state errors across all batches and steps
    # Shape tracked: (N_samples, H, state_dim) — we keep running sums for efficiency
    per_state_abs_err_sum  = np.zeros(state_dim)   # for MAE
    per_state_sq_err_sum   = np.zeros(state_dim)   # for RMSE
    per_state_abs_gt_sum   = np.zeros(state_dim)   # mean |ground truth| for relative error
    per_state_count        = 0                      # total (samples × steps)

    # For per-horizon error: store lists of errors and gts for each horizon step
    horizon_steps = [1, 3, 5, 10]
    horizon_steps = [h for h in horizon_steps if h <= horizon]
    per_horizon_err = {h: [] for h in horizon_steps}
    per_horizon_gt = {h: [] for h in horizon_steps}

    for batch_idx, batch_data in enumerate(loader):
        batch_initial = batch_data['initial_states'].to(device)
        batch_actions = batch_data['actions'].to(device)
        batch_states  = batch_data['states'].to(device)

        predicted_states = []
        residual_norms   = []
        current_state    = batch_initial

        # Only pass denorm/min_bounds to the ODE when normalize=True — the ODE uses
        # these to convert between normalized and raw space internally. When normalize=False
        # the ODE already operates in raw space and must not receive these tensors.
        ode_module = HybridDynamicsODE(
            hybrid_model, device,
            denorm_factors=denorm_factors_torch if config['data']['normalize'] else None,
            min_bounds=min_bounds_torch if config['data']['normalize'] else None,
        ).to(device)

        for step in range(horizon):
            action = batch_actions[:, step, :]

            # Only compute residual regularization if model uses residual (modular ablation)
            if hybrid_model.with_residual:
                residual_output = hybrid_model.residual_network(current_state, action)
                
                # Denormalize residual to raw space before penalizing (if using normalization)
                # This ensures regularization penalty is consistent with trajectory loss (both in raw space)
                if config['data']['normalize']:
                    residual_output_raw = residual_output * denorm_factors_torch
                    residual_norm = torch.norm(residual_output_raw, p=2, dim=1).mean()
                else:
                    residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
                residual_norms.append(residual_norm)

            ode_module.set_action(action)
            t_eval   = torch.tensor([0.0, dt], dtype=current_state.dtype, device=device)
            if config['data']['normalize']:
                current_state_raw = (current_state + 1.0) * denorm_factors_torch + min_bounds_torch
            else:
                current_state_raw = current_state
            solution = odeint(ode_module, current_state_raw, t_eval,
                              method=ode_method, rtol=ode_rtol, atol=ode_atol)
            next_state_raw = solution[-1].clamp(-100.0, 100.0)
            if config['data']['normalize']:
                next_state = 2.0 * (next_state_raw - min_bounds_torch) / (2.0 * denorm_factors_torch) - 1.0
            else:
                next_state = next_state_raw
            predicted_states.append(next_state)
            current_state = next_state

        predicted_trajectory = torch.stack(predicted_states, dim=1)  # (B, H, state_dim)
        
        # Denormalize predictions and ground truth to raw space (same as training)
        if config['data']['normalize']:
            predicted_trajectory_raw = (predicted_trajectory + 1.0) * denorm_factors_torch + min_bounds_torch
            batch_states_raw = (batch_states + 1.0) * denorm_factors_torch + min_bounds_torch
        else:
            predicted_trajectory_raw = predicted_trajectory
            batch_states_raw = batch_states
        
        prediction_error = predicted_trajectory_raw - batch_states_raw

        # Capture the raw-space error (physical units) BEFORE any loss scaling.
        # This is used exclusively for the per-state reporting table (MAE/RMSE/rel err)
        # so that those numbers are always interpretable regardless of the loss flags.
        prediction_error_raw = prediction_error.detach()

        # Apply per_state_loss_norm scaling only for the loss value itself, mirroring
        # exactly what the training objective does.
        # Only applied if per_state_scales was computed and passed in
        if per_state_scales is not None:
            prediction_error = prediction_error / (per_state_scales ** 2)

        # --- Compounded trajectory loss (what training optimizes) ---
        traj_loss  = torch.norm(prediction_error, p=2, dim=2).mean().item()
        # Regularization loss is only non-zero if model uses residual (modular ablation)
        reg_loss   = torch.stack(residual_norms).mean().item() if residual_norms else 0.0
        total_loss = reg_loss + lambda_val * traj_loss

        all_traj_losses.append(traj_loss)
        all_reg_losses.append(reg_loss)
        all_total_losses.append(total_loss)

        # --- Per-state error accumulation (compounded errors from full trajectory) ---
        # Always use raw-space error so MAE/RMSE are in physical units (rad, m/s, …)
        # regardless of normalize / per_state_loss_norm flags.
        err_np   = prediction_error_raw.cpu().numpy()    # (B, H, state_dim) — raw space
        gt_np    = batch_states_raw.cpu().numpy()        # (B, H, state_dim) — raw space
        err_flat = err_np.reshape(-1, state_dim)         # (B*H, state_dim)
        gt_flat  = gt_np.reshape(-1, state_dim)          # (B*H, state_dim)
        per_state_abs_err_sum += np.abs(err_flat).sum(axis=0)
        per_state_sq_err_sum  += (err_flat ** 2).sum(axis=0)
        per_state_abs_gt_sum  += np.abs(gt_flat).sum(axis=0)
        per_state_count       += err_flat.shape[0]

        # --- Per-horizon compounded error accumulation (also raw space) ---
        for h in horizon_steps:
            if h-1 < err_np.shape[1]:
                per_horizon_err[h].append(err_np[:, h-1, :])  # (B, state_dim)
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
    # Per-state accuracy table (with per-horizon rel err)
    # -------------------------------------------------------------------------
    per_state_mae      = per_state_abs_err_sum / per_state_count
    per_state_rmse     = np.sqrt(per_state_sq_err_sum / per_state_count)
    per_state_mean_mag = per_state_abs_gt_sum / per_state_count          # mean |gt| per state
    # Relative error: MAE expressed as % of the mean observed magnitude.
    # Where the mean magnitude is very small (< 1e-6), mark as undefined.
    per_state_rel_err  = np.where(
        per_state_mean_mag > 1e-6,
        100.0 * per_state_mae / per_state_mean_mag,
        np.nan,
    )

    # Compute per-horizon relative errors
    per_horizon_rel_err = {}
    for h in horizon_steps:
        if per_horizon_err[h]:
            err_h = np.concatenate(per_horizon_err[h], axis=0)  # (N, state_dim)
            gt_h  = np.concatenate(per_horizon_gt[h], axis=0)
            mae_h = np.abs(err_h).mean(axis=0)
            mean_mag_h = np.abs(gt_h).mean(axis=0)
            rel_err_h = np.where(mean_mag_h > 1e-6, 100.0 * mae_h / mean_mag_h, np.nan)
            per_horizon_rel_err[h] = rel_err_h
        else:
            per_horizon_rel_err[h] = np.full(state_dim, np.nan)

    print(f"\n{'='*72}")
    print(f"PER-STATE ACCURACY  [{split_name.upper()}]")
    print(f"  COMPOUNDED ERRORS: Predictions use previous predictions (how loss is computed)")
    print(f"  (averaged over all {per_state_count} predictions = batches × horizon)")
    print(f"  Relative error = MAE / mean(|ground truth|)")
    print(f"{'='*72}")
    # Table header
    header = f"  {'State':<22}  {'MAE':>10}  {'RMSE':>10}  {'Mean |gt|':>10}  {'Rel err %':>10}"
    for h in horizon_steps:
        header += f"  [h={h}]%"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(STATE_NAMES):
        rel_str = f"{per_state_rel_err[i]:>9.2f}%" if not np.isnan(per_state_rel_err[i]) else "       N/A"
        row = f"  {name:<22}  {per_state_mae[i]:>10.6f}  {per_state_rmse[i]:>10.6f}  {per_state_mean_mag[i]:>10.4f}  {rel_str}"
        for h in horizon_steps:
            rel_h = per_horizon_rel_err[h][i]
            rel_h_str = f"{rel_h:>8.2f}%" if not np.isnan(rel_h) else "    N/A"
            row += f"  {rel_h_str}"
        print(row)
    print(f"{'='*72}\n")

    return {
        'mean_traj_loss':    np.mean(all_traj_losses),
        'std_traj_loss':     np.std(all_traj_losses),
        'mean_reg_loss':     np.mean(all_reg_losses),
        'mean_total_loss':   np.mean(all_total_losses),
        'per_state_mae':     per_state_mae,
        'per_state_rmse':    per_state_rmse,
        'per_state_mean_mag': per_state_mean_mag,
        'per_state_rel_err': per_state_rel_err,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained hybrid physics-augmented world model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint (training checkpoint or final_model.pt)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which data split to evaluate on (default: test)')
    parser.add_argument('--with-prior', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='Include physics prior in model (default: True)')
    parser.add_argument('--with-residual', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='Include learned residual in model (default: True)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("HYBRID PHYSICS-AUGMENTED MODEL — EVALUATION")
    print("="*60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print(f"  Physics prior : {args.with_prior}")
    print(f"  Residual network : {args.with_residual}")

    # Load config from the physics directory (same as training script)
    config_path = Path(__file__).parent / 'training_params.yaml'
    print(f"  Config     : {config_path}")
    config = load_config(str(config_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {device}")

    # Load data
    csv_path = Path(__file__).parent.parent / "data" / "updated_trajectory_data_progressive_noatmo.csv"
    train_loader, val_loader, test_loader, denorm_factors, min_bounds = load_trajectory_data(str(csv_path), config)
    denorm_factors_torch = torch.tensor(denorm_factors, dtype=torch.float32, device=device)
    min_bounds_torch = torch.tensor(min_bounds, dtype=torch.float32, device=device)
    
    # Compute actual scales from data ONLY if per-state loss scaling is enabled
    # These are applied differently from denorm_factors:
    # - denorm_factors: used for state denormalization in ODE integrator (when normalize=true)
    # - per_state_scales: used for loss computation scaling (only if per_state_loss_norm=true)
    per_state_scales_torch = None
    if config['data'].get('per_state_loss_norm', False):
        print("\n  Computing actual per-state scales from training data (for consistent evaluation)...")
        df = pd.read_csv(str(csv_path))
        actual_scales = compute_actual_scales(df)
        per_state_scales_torch = torch.tensor(actual_scales, dtype=torch.float32, device=device)
        print(f"  Using actual scales for loss computation: {actual_scales}")
    else:
        print("\n  per_state_loss_norm disabled - evaluating with raw loss (no scaling)")

    loader_map = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    target_loader = loader_map[args.split]

    # Build model and load weights
    hybrid_model = initialize_models(config, args.checkpoint, device,
                                     with_prior=args.with_prior,
                                     with_residual=args.with_residual)

    # Run evaluation
    # denorm_factors is always passed — it is needed for ODE state denormalization when normalize=True.
    # min_bounds is only needed when normalize=True (state de/renorm in ODE).
    # per_state_scales is passed when per_state_loss_norm was enabled during training.
    evaluate(hybrid_model, target_loader, config, device, split_name=args.split,
             denorm_factors=denorm_factors_torch,
             min_bounds=min_bounds_torch if config['data']['normalize'] else None,
             per_state_scales=per_state_scales_torch)

'''
run script like this for exmaple:
 python test_physics_model.py --checkpoint checkpoints/exp_v0.6/epoch_10.pt
'''
if __name__ == '__main__':
    main()