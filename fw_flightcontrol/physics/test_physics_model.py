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

# Add project to path
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import HybridDynamicsODE


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

    if normalize:
        raw_states = df[state_cols].values.copy()
        raw_states[:, 2] = raw_states[:, 2] / 3.6
        state_mean = raw_states.mean(axis=0)
        state_std  = raw_states.std(axis=0) + 1e-8
        print(f"State normalization: mean={state_mean[:3]}..., std={state_std[:3]}...")
    else:
        state_mean = np.zeros(state_dim)
        state_std  = np.ones(state_dim)

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

            if normalize:
                seq_states      = (seq_states - state_mean) / state_std
                seq_next_states = (seq_next_states - state_mean) / state_std

            traj_sequences.append({
                'initial_state': seq_states[0].copy(),
                'actions':       seq_actions.copy(),
                'states':        seq_next_states.copy(),
            })

        trajectory_sequences_by_traj[traj_id] = traj_sequences

    trajectories = sorted(list(df['trajectory_id'].unique()))
    np.random.seed(42)  # Fixed seed for reproducible splits
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

    return make_loader(train_seqs), make_loader(val_seqs), make_loader(test_seqs)


def initialize_models(config: Dict, checkpoint_path: str, device: torch.device) -> HybridDynamicsModel:
    """
    Build the hybrid model and load residual network weights from checkpoint.
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
        with_prior=True,
        with_residual=True,
        integration_method=integration_method,
    )
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()

    print(f"  ✓ Hybrid model ready on {device}")
    return hybrid_model


@torch.no_grad()
def evaluate(hybrid_model: HybridDynamicsModel, loader, config: Dict, device: torch.device,
             split_name: str = "eval") -> Dict:
    """
    Run one evaluation epoch over the given DataLoader.
    Prints per-batch statistics and a final summary (including per-state accuracy) to the terminal.
    """
    from torchdiffeq import odeint

    horizon    = config['training']['horizon']
    ode_method = config['integration']['method']
    ode_rtol   = config['integration']['rtol']
    ode_atol   = config['integration']['atol']
    dt         = config['integration']['dt']
    lambda_val = config['aphynity'].get('lambda_init', 1.0)
    state_dim  = config['network']['state_dim']

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

    for batch_idx, batch_data in enumerate(loader):
        batch_initial = batch_data['initial_states'].to(device)
        batch_actions = batch_data['actions'].to(device)
        batch_states  = batch_data['states'].to(device)

        predicted_states = []
        residual_norms   = []
        current_state    = batch_initial

        ode_module = HybridDynamicsODE(hybrid_model, device).to(device)

        for step in range(horizon):
            action = batch_actions[:, step, :]

            residual_output = hybrid_model.residual_network(current_state, action)
            residual_norm   = torch.norm(residual_output, p=2, dim=1).mean()
            residual_norms.append(residual_norm)

            ode_module.set_action(action)
            t_eval   = torch.tensor([0.0, dt], dtype=current_state.dtype, device=device)
            solution = odeint(ode_module, current_state, t_eval,
                              method=ode_method, rtol=ode_rtol, atol=ode_atol)
            next_state = solution[-1].clamp(-100.0, 100.0)
            predicted_states.append(next_state)
            current_state = next_state

        predicted_trajectory = torch.stack(predicted_states, dim=1)  # (B, H, state_dim)
        prediction_error     = predicted_trajectory - batch_states    # (B, H, state_dim)

        # --- Global trajectory loss (L2 norm across state dim, mean over B and H) ---
        traj_loss  = torch.norm(prediction_error, p=2, dim=2).mean().item()
        reg_loss   = torch.stack(residual_norms).mean().item()
        total_loss = reg_loss + lambda_val * traj_loss

        all_traj_losses.append(traj_loss)
        all_reg_losses.append(reg_loss)
        all_total_losses.append(total_loss)

        # --- Per-state error accumulation (over B and H jointly) ---
        err_np   = prediction_error.cpu().numpy()        # (B, H, state_dim)
        gt_np    = batch_states.cpu().numpy()            # (B, H, state_dim)
        err_flat = err_np.reshape(-1, state_dim)         # (B*H, state_dim)
        gt_flat  = gt_np.reshape(-1, state_dim)          # (B*H, state_dim)
        per_state_abs_err_sum += np.abs(err_flat).sum(axis=0)
        per_state_sq_err_sum  += (err_flat ** 2).sum(axis=0)
        per_state_abs_gt_sum  += np.abs(gt_flat).sum(axis=0)
        per_state_count       += err_flat.shape[0]

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
    # Per-state accuracy table
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

    print(f"\n{'='*72}")
    print(f"PER-STATE ACCURACY  [{split_name.upper()}]")
    print(f"  (averaged over all {per_state_count} predictions = batches × horizon)")
    print(f"  Relative error = MAE / mean(|ground truth|) — how large the error is")
    print(f"  compared to the typical magnitude of that state variable.")
    print(f"{'='*72}")
    header = f"  {'State':<22}  {'MAE':>10}  {'RMSE':>10}  {'Mean |gt|':>10}  {'Rel err %':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(STATE_NAMES):
        rel_str = f"{per_state_rel_err[i]:>9.2f}%" if not np.isnan(per_state_rel_err[i]) else "       N/A"
        print(f"  {name:<22}  {per_state_mae[i]:>10.6f}  {per_state_rmse[i]:>10.6f}"
              f"  {per_state_mean_mag[i]:>10.4f}  {rel_str}")
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
    args = parser.parse_args()

    print("\n" + "="*60)
    print("HYBRID PHYSICS-AUGMENTED MODEL — EVALUATION")
    print("="*60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")

    # Load config from the physics directory (same as training script)
    config_path = Path(__file__).parent / 'training_params.yaml'
    print(f"  Config     : {config_path}")
    config = load_config(str(config_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {device}")

    # Load data
    csv_path = Path(__file__).parent.parent / "data" / "updated_trajectory_data_noatmo.csv"
    train_loader, val_loader, test_loader = load_trajectory_data(str(csv_path), config)

    loader_map = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    target_loader = loader_map[args.split]

    # Build model and load weights
    hybrid_model = initialize_models(config, args.checkpoint, device)

    # Run evaluation
    evaluate(hybrid_model, target_loader, config, device, split_name=args.split)

'''
run script like this for exmaple:
 python test_physics_model.py --checkpoint checkpoints/exp_v0.4/epoch_520.pt
'''
if __name__ == '__main__':
    main()