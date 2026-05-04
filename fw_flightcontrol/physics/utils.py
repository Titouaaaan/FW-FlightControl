"""Shared utilities for physics model training and evaluation."""

import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STATE_INDICES = [0, 1, 2, 3, 4, 5, 8, 9]

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


# ============================================================================
# CONFIG
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# NORMALIZATION
# ============================================================================

def get_norm_type(config: Dict) -> Optional[str]:
    """Return the normalization type string, or None if normalization is disabled.

    Reads `data.normalization_type` first. Falls back to the legacy `data.normalize`
    boolean (True → 'bounds_normalization') for backward compatibility.

    Supported values: 'bounds_normalization', 'data_driven_normalization', None.
    """
    norm_type = config['data'].get('normalization_type')
    if norm_type is not None:
        return norm_type
    if config['data'].get('normalize', False):
        return 'bounds_normalization'
    return None


def extract_bounds_from_config(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract state bounds from config as (min_bounds, max_bounds) arrays of shape (state_dim,).

    Order: [roll, pitch, airspeed_mps, p, q, r, alpha, beta]
    """
    bounds = config['state_bounds']
    keys = ['roll', 'pitch', 'airspeed_mps', 'p', 'q', 'r', 'alpha', 'beta']
    min_bounds = np.array([bounds[f'{k}_min'] for k in keys], dtype=np.float32)
    max_bounds = np.array([bounds[f'{k}_max'] for k in keys], dtype=np.float32)
    return min_bounds, max_bounds


def compute_denorm_factors(min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
    """Return (max - min) / 2 per state — the scale for bounds normalization."""
    return (max_bounds - min_bounds) / 2.0


def compute_data_norm_params(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-state mean and std from a (training) dataframe.

    Returns:
        (mean, std): both of shape (state_dim,), computed on raw physical units.
        std has a small epsilon added to avoid division by zero.
    """
    state_cols = [f's_t_{i}' for i in STATE_INDICES]
    means, stds = [], []
    for col in state_cols:
        values = df[col].values.copy()
        if col == 's_t_2':  # airspeed: km/h → m/s
            values = values / 3.6
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)) + 1e-8)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def compute_actual_scales(df: pd.DataFrame) -> np.ndarray:
    """Compute per-state mean |value| from training data, used for per-state loss scaling."""
    state_cols = [f's_t_{i}' for i in STATE_INDICES]
    scales = []
    for col in state_cols:
        values = df[col].values.copy()
        if col == 's_t_2':  # airspeed: km/h → m/s
            values = values / 3.6
        scales.append(np.mean(np.abs(values)))
    return np.array(scales, dtype=np.float32)


def normalize_state_np(
    state: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    norm_type: str,
) -> np.ndarray:
    """Normalize a numpy state array.

    bounds_normalization:      s_norm = (s - offset) / scale - 1   (offset=min, scale=(max-min)/2)
    data_driven_normalization: s_norm = (s - offset) / scale        (offset=mean, scale=std)
    """
    if norm_type == 'bounds_normalization':
        return (state - offset) / scale - 1.0
    if norm_type == 'data_driven_normalization':
        return (state - offset) / scale
    raise ValueError(f"Unknown normalization type: {norm_type!r}")


def normalize_state_torch(
    state: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    norm_type: str,
) -> torch.Tensor:
    """Normalize a torch state tensor.

    bounds_normalization:      s_norm = (s - offset) / scale - 1
    data_driven_normalization: s_norm = (s - offset) / scale
    """
    if norm_type == 'bounds_normalization':
        return (state - offset) / scale - 1.0
    if norm_type == 'data_driven_normalization':
        return (state - offset) / scale
    raise ValueError(f"Unknown normalization type: {norm_type!r}")


def denormalize_state_torch(
    state: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    norm_type: str,
) -> torch.Tensor:
    """Denormalize a torch state tensor (inverse of normalize_state_torch).

    bounds_normalization:      s_raw = (s_norm + 1) * scale + offset
    data_driven_normalization: s_raw = s_norm * scale + offset
    """
    if norm_type == 'bounds_normalization':
        return (state + 1.0) * scale + offset
    if norm_type == 'data_driven_normalization':
        return state * scale + offset
    raise ValueError(f"Unknown normalization type: {norm_type!r}")


# ============================================================================
# DATA LOADING
# ============================================================================

class TrajectoryDataset(torch.utils.data.Dataset):
    """Dataset wrapping trajectory sequences as fixed-length sliding windows."""

    def __init__(self, sequences: List[Dict]):
        # Convert numpy arrays to tensors once at construction; __getitem__ returns pre-built tensors
        self.sequences = [
            {
                'initial_states': torch.tensor(seq['initial_state'], dtype=torch.float32),
                'actions':        torch.tensor(seq['actions'],        dtype=torch.float32),
                'states':         torch.tensor(seq['states'],         dtype=torch.float32),
            }
            for seq in sequences
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        return self.sequences[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        return {
            'initial_states': torch.stack([b['initial_states'] for b in batch]),
            'actions':        torch.stack([b['actions']        for b in batch]),
            'states':         torch.stack([b['states']         for b in batch]),
        }


def load_trajectory_data(csv_path: str, config: Dict) -> Tuple:
    """Load trajectory CSV and return (train_loader, val_loader, test_loader, norm_scale, norm_offset).

    norm_scale and norm_offset encode the normalization parameters:
      - bounds_normalization:      norm_scale = (max-min)/2, norm_offset = min
      - data_driven_normalization: norm_scale = std,         norm_offset = mean
      - no normalization:          norm_scale = (max-min)/2, norm_offset = min  (unused)

    For data-driven normalization the statistics are computed from the training
    split only to avoid leakage into validation/test data.
    """
    print(f"\nLoading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transitions from {df['trajectory_id'].nunique()} trajectories")

    horizon    = config['training']['horizon']
    action_dim = config['network']['action_dim']
    batch_size = config['training']['batch_size']

    state_cols      = [f's_t_{i}'   for i in STATE_INDICES]
    action_cols     = [f'a_t_{i}'   for i in range(action_dim)]
    next_state_cols = [f's_t+1_{i}' for i in STATE_INDICES]

    # ── Determine train/val/test split first ─────────────────────────────────
    # (needed before computing data-driven norm params from training data only)
    trajectories = sorted(df['trajectory_id'].unique().tolist())
    np.random.seed(config['data'].get('random_seed', 42))
    np.random.shuffle(trajectories)

    n_train = int(len(trajectories) * config['data']['train_fraction'])
    n_val   = int(len(trajectories) * config['data']['val_fraction'])

    train_ids = set(trajectories[:n_train])
    val_ids   = set(trajectories[n_train:n_train + n_val])
    test_ids  = set(trajectories[n_train + n_val:])

    # ── Normalization parameters ──────────────────────────────────────────────
    norm_type = get_norm_type(config)
    min_bounds, max_bounds = extract_bounds_from_config(config)

    if norm_type == 'data_driven_normalization':
        train_df = df[df['trajectory_id'].isin(train_ids)]
        norm_offset, norm_scale = compute_data_norm_params(train_df)
        print(f"\nData-driven normalization (computed from {len(train_df)} training rows):")
        print(f"  Mean (norm_offset): {norm_offset}")
        print(f"  Std  (norm_scale):  {norm_scale}")
    elif norm_type == 'bounds_normalization':
        norm_offset = min_bounds
        norm_scale  = compute_denorm_factors(min_bounds, max_bounds)
        print(f"\nBounds normalization (from config):")
        print(f"  Min (norm_offset): {norm_offset}")
        print(f"  Max:               {max_bounds}")
        print(f"  Scale (max-min)/2: {norm_scale}")
    else:
        # No normalization — return bounds params for reference (used by ODE but not for norm)
        norm_offset = min_bounds
        norm_scale  = compute_denorm_factors(min_bounds, max_bounds)

    # ── Build sequences ───────────────────────────────────────────────────────
    trajectory_sequences_by_traj: Dict = {}

    for traj_id, group in df.groupby('trajectory_id'):
        group = group.sort_values('step_id').reset_index(drop=True)

        states      = group[state_cols].values.copy()
        actions     = group[action_cols].values
        next_states = group[next_state_cols].values.copy()

        states[:, 2]      = states[:, 2] / 3.6   # km/h → m/s
        next_states[:, 2] = next_states[:, 2] / 3.6

        traj_sequences = []
        for start_idx in range(len(states) - horizon):
            seq_states      = states[start_idx:start_idx + horizon]
            seq_actions     = actions[start_idx:start_idx + horizon]
            seq_next_states = next_states[start_idx:start_idx + horizon]

            if norm_type is not None:
                seq_states      = normalize_state_np(seq_states,      norm_scale, norm_offset, norm_type)
                seq_next_states = normalize_state_np(seq_next_states, norm_scale, norm_offset, norm_type)

            traj_sequences.append({
                'initial_state': seq_states[0].copy(),
                'actions':       seq_actions.copy(),
                'states':        seq_next_states.copy(),
            })

        trajectory_sequences_by_traj[traj_id] = traj_sequences

    def build_seqs(ids):
        return [seq for tid, seqs in trajectory_sequences_by_traj.items()
                if tid in ids for seq in seqs]

    train_seqs = build_seqs(train_ids)
    val_seqs   = build_seqs(val_ids)
    test_seqs  = build_seqs(test_ids)

    print(f"\nTrain/Val/Test split (trajectory level):")
    print(f"  Trajectories: {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test")
    print(f"  Sequences:    {len(train_seqs)} train | {len(val_seqs)} val | {len(test_seqs)} test")

    def make_loader(seqs, shuffle=False):
        return torch.utils.data.DataLoader(
            TrajectoryDataset(seqs),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=TrajectoryDataset.collate_fn,
        )

    # norm_scale / norm_offset replace the old denorm_factors / min_bounds in callers.
    # Their meaning depends on norm_type; callers must also propagate norm_type.
    return make_loader(train_seqs, shuffle=True), make_loader(val_seqs), make_loader(test_seqs), norm_scale, norm_offset


# ============================================================================
# LOGGING
# ============================================================================

def log_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_metrics: Dict,
    val_metrics: Optional[Dict] = None,
    lambda_current: float = 0.0,
) -> None:
    """Print a single-line epoch summary to stdout."""
    parts = [
        f"Epoch {epoch+1:3d}/{num_epochs}",
        f"L_total={train_metrics['loss_total']:.4f}",
        f"L_traj={train_metrics['loss_trajectory']:.4f}",
        f"L_reg={train_metrics['loss_regularization']:.4f}",
        f"λ={lambda_current:.4f}",
    ]
    if val_metrics is not None:
        parts.append(f"Val_L={val_metrics['loss_total']:.4f}")
    print(" | ".join(parts))


def log_tensorboard_epoch(
    writer,
    epoch: int,
    train_metrics: Dict,
    val_metrics: Optional[Dict] = None,
    lambda_current: float = 0.0,
    grad_metrics: Optional[Dict] = None,
    current_lr: Optional[float] = None,
) -> None:
    """Write all epoch-level scalars to TensorBoard in a single call."""
    writer.add_scalar('Epoch/train_loss_total',          train_metrics['loss_total'],         epoch)
    writer.add_scalar('Epoch/train_loss_trajectory',     train_metrics['loss_trajectory'],     epoch)
    writer.add_scalar('Epoch/train_loss_regularization', train_metrics['loss_regularization'], epoch)
    writer.add_scalar('Epoch/lambda_final',              lambda_current,                       epoch)

    if val_metrics is not None:
        writer.add_scalar('Epoch/val_loss_total',          val_metrics['loss_total'],          epoch)
        writer.add_scalar('Epoch/val_loss_trajectory',     val_metrics['loss_trajectory'],      epoch)
        writer.add_scalar('Epoch/val_loss_regularization', val_metrics['loss_regularization'],  epoch)

    if grad_metrics is not None:
        writer.add_scalar('Gradients/norm_before_clipping', grad_metrics['grad_norm_before_clipping'], epoch)
        writer.add_scalar('Gradients/norm_after_clipping',  grad_metrics['grad_norm_after_clipping'],  epoch)

    if current_lr is not None:
        writer.add_scalar('Training/learning_rate', current_lr, epoch)


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(
    path,
    epoch: int,
    hybrid_model,
    optimizer,
    scheduler,
    lambda_current: float,
    train_history: Dict,
    val_history: Dict,
    norm_scale: Optional[np.ndarray] = None,
    norm_offset: Optional[np.ndarray] = None,
) -> None:
    """Save a training checkpoint to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch':           epoch,
        'residual_state':  hybrid_model.residual_network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lambda':          lambda_current,
        'train_history':   train_history,
        'val_history':     val_history,
    }
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    if norm_scale is not None:
        checkpoint['norm_scale']  = norm_scale.tolist()
        checkpoint['norm_offset'] = norm_offset.tolist()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
