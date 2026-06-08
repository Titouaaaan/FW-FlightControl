"""Shared utilities for physics model training and evaluation."""

import os
import sys
import csv
import torch
import yaml
import numpy as np
import pandas as pd
from contextlib import contextmanager
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


def clean_state_dict_for_compilation(state_dict: Dict) -> Dict:
    """
    Remove torch.compile artifacts from state dict.
    
    When models are compiled with torch.compile, the state dict keys are prefixed
    with '_orig_mod.'. This function strips those prefixes so the state dict can
    be loaded into an uncompiled model.
    
    Args:
        state_dict: State dict potentially containing '_orig_mod.' prefixes
    
    Returns:
        Cleaned state dict with prefixes removed (if present)
    """
    # Check if any key has the _orig_mod prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  ⚠ Detected torch.compile artifacts in state dict, cleaning...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    return state_dict


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
    arch_config: Optional[Dict] = None,
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
    if arch_config is not None:
        checkpoint['arch_config'] = {
            'activation':   arch_config.get('activation', 'relu'),
            'hidden_dims':  arch_config.get('hidden_dims', []),
            'state_dim':    arch_config.get('state_dim', 8),
            'action_dim':   arch_config.get('action_dim', 3),
            'use_batch_norm': arch_config.get('use_batch_norm', False),
        }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


# ============================================================================
# CONTROL EVALUATION UTILITIES
# ============================================================================

def compute_convergence_stats(
    history_deg: List[float],
    target_deg: float,
    threshold_deg: float = 1.0,
    min_stable_steps: int = 100,
) -> Dict:
    """Detect convergence and measure steady-state stability.

    Convergence is defined as the first step after which the error stays
    within `threshold_deg` for at least `min_stable_steps` consecutive steps.
    Steady-state statistics are computed from that point onward (or the last
    20 % of steps if convergence is never reached).

    Returns a dict with:
      convergence_step  — step index of first convergence, or None
      steady_mean_error — mean |angle - target| in steady state [°]
      steady_std        — std of actual angle in steady state [°] (oscillation measure)
    """
    errors = np.abs(np.array(history_deg) - target_deg)
    n = len(errors)

    convergence_step = None
    for i in range(n - min_stable_steps + 1):
        if np.all(errors[i : i + min_stable_steps] < threshold_deg):
            convergence_step = i
            break

    steady_slice = slice(convergence_step, None) if convergence_step is not None \
                   else slice(int(0.8 * n), None)

    return {
        'convergence_step': convergence_step,
        'steady_mean_error': float(np.mean(errors[steady_slice])),
        'steady_std': float(np.std(np.array(history_deg)[steady_slice])),
    }


# ============================================================================
# JSBSim ENV STATE SAVE / RESTORE
# ============================================================================

def get_env_state(env) -> dict:
    """Snapshot the physical state of a JSBSim env for later restoration.

    Captures attitude, airspeed, angular rates, altitude and throttle — the
    variables that fully determine short-horizon dynamics.  Alpha/beta are
    implicitly encoded in the body-velocity components (u, v, w).
    """
    from fw_jsbgym.utils import jsbsim_properties as prp
    sim = env.unwrapped.sim
    return {
        'roll_rad':              float(sim[prp.roll_rad]),
        'pitch_rad':             float(sim[prp.pitch_rad]),
        'heading_rad':           float(sim[prp.heading_rad]),
        'airspeed_kts':          float(sim[prp.airspeed_kts]),
        'p_radps':               float(sim[prp.p_radps]),
        'q_radps':               float(sim[prp.q_radps]),
        'r_radps':               float(sim[prp.r_radps]),
        'altitude_ft':           float(sim[prp.altitude_sl_ft]),
        'throttle':              float(sim[prp.throttle_cmd]),
        'alpha_rad':             float(sim.fdm['aero/alpha-rad']),
        'beta_rad':              float(sim.fdm['aero/beta-rad']),
        'aileron_cmd':           float(sim[prp.aileron_cmd]),
        'elevator_cmd':          float(sim[prp.elevator_cmd]),
        # Actual FCS surface positions (after actuator lag dynamics).
        # Must be set BEFORE run_ic() so JSBSim's force evaluation uses the
        # correct actuation state and stores the right prev-acceleration in its
        # internal multi-step integrator buffers (dqUVWidot / dqPQRidot).
        'fcs_aileron_pos_rad':   float(sim.fdm['fcs/left-aileron-pos-rad']),
        'fcs_elevator_pos_rad':  float(sim.fdm['fcs/elevator-pos-rad']),
        'fcs_throttle_pos_norm': float(sim.fdm['fcs/throttle-pos-norm']),
    }


def set_env_state(env, state: dict) -> None:
    """Restore a JSBSim env to a previously snapshotted state.

    Writes all physical state into JSBSim's IC slots, then sets throttle,
    control commands, and actual FCS surface positions BEFORE calling run_ic().
    This ensures the IC force evaluation uses the correct actuation state so
    that JSBSim's internal multi-step integrator history buffers
    (dqUVWidot / dqPQRidot) are populated with the right prev-acceleration.
    Without this, the first post-restore step diverges because the stored
    acceleration was computed at trim rather than at s0.
    """
    from fw_jsbgym.utils import jsbsim_properties as prp
    sim = env.unwrapped.sim

    # IC attitude / velocity / position
    sim[prp.ic_roll_rad]     = state['roll_rad']
    sim[prp.ic_pitch_rad]    = state['pitch_rad']   - state['alpha_rad']
    sim[prp.ic_heading_rad]  = state['heading_rad'] + state['beta_rad']
    sim[prp.ic_airspeed_kts] = state['airspeed_kts']
    sim[prp.ic_p_radps]      = state['p_radps']
    sim[prp.ic_q_radps]      = state['q_radps']
    sim[prp.ic_r_radps]      = state['r_radps']
    sim[prp.ic_altitude_ft]  = state['altitude_ft']
    sim.fdm['ic/alpha-rad']  = state['alpha_rad']
    sim.fdm['ic/beta-rad']   = state['beta_rad']

    # Set actuation state BEFORE run_ic() so the IC force evaluation uses s0
    # values and stores the correct prev-acceleration in the integrator deques.
    sim[prp.throttle_cmd]  = state['throttle']
    sim[prp.aileron_cmd]   = state['aileron_cmd']
    sim[prp.elevator_cmd]  = state['elevator_cmd']
    sim.fdm['fcs/left-aileron-pos-rad']  = state.get('fcs_aileron_pos_rad', 0.0)
    sim.fdm['fcs/right-aileron-pos-rad'] = state.get('fcs_aileron_pos_rad', 0.0)
    sim.fdm['fcs/elevator-pos-rad']      = state.get('fcs_elevator_pos_rad', 0.0)
    sim.fdm['fcs/throttle-pos-norm']     = state.get('fcs_throttle_pos_norm', state['throttle'])

    sim.fdm.run_ic()

    # Restore commands after run_ic() (run_ic may reset some cmd properties)
    sim[prp.throttle_cmd]  = state['throttle']
    sim[prp.aileron_cmd]   = state['aileron_cmd']
    sim[prp.elevator_cmd]  = state['elevator_cmd']


def plot_tracking_performance(
    target_roll: float,
    target_pitch: float,
    filename: str = 'tracking_performance.png',
    mppi_roll: Optional[List[float]] = None,
    mppi_pitch: Optional[List[float]] = None,
    pid_roll: Optional[List[float]] = None,
    pid_pitch: Optional[List[float]] = None,
    dt: float = 0.01,
) -> None:
    """Save a 2-subplot tracking performance figure (roll top, pitch bottom).

    Always saves to FW-FlightControl/fw_flightcontrol/data/mppi-performance/,
    creating the folder if it does not exist.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_dir = Path(__file__).parent.parent / 'data' / 'mppi-performance'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    fig, (ax_roll, ax_pitch) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    def time_axis(history):
        return np.arange(len(history)) * dt

    # --- Roll subplot ---
    if mppi_roll is not None:
        t = time_axis(mppi_roll)
        ax_roll.plot(t, mppi_roll, color='steelblue', linewidth=1.2, label='MPPI')
        ax_roll.axhline(target_roll, color='red', linestyle='--', linewidth=1.2, label='Target')
    if pid_roll is not None:
        t = time_axis(pid_roll)
        ax_roll.plot(t, pid_roll, color='darkorange', linewidth=1.2, label='PID')
        if mppi_roll is None:
            ax_roll.axhline(target_roll, color='red', linestyle='--', linewidth=1.2, label='Target')
    ax_roll.set_ylabel('Roll angle [°]')
    ax_roll.legend(loc='upper right')
    ax_roll.grid(True, alpha=0.3)

    # --- Pitch subplot ---
    if mppi_pitch is not None:
        t = time_axis(mppi_pitch)
        ax_pitch.plot(t, mppi_pitch, color='steelblue', linewidth=1.2, label='MPPI')
        ax_pitch.axhline(target_pitch, color='red', linestyle='--', linewidth=1.2, label='Target')
    if pid_pitch is not None:
        t = time_axis(pid_pitch)
        ax_pitch.plot(t, pid_pitch, color='darkorange', linewidth=1.2, label='PID')
        if mppi_pitch is None:
            ax_pitch.axhline(target_pitch, color='red', linestyle='--', linewidth=1.2, label='Target')
    ax_pitch.set_ylabel('Pitch angle [°]')
    ax_pitch.set_xlabel('Time [s]')
    ax_pitch.legend(loc='upper right')
    ax_pitch.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Tracking plot saved to {save_path}")


def plot_action_history(
    filename: str = 'action_history.png',
    mppi_actions: Optional[np.ndarray] = None,
    pid_actions: Optional[np.ndarray] = None,
    dt: float = 0.01,
) -> None:
    """Save a 3-subplot action history figure (aileron, elevator, throttle).

    mppi_actions / pid_actions: arrays of shape (N, 3) — columns are
    [aileron, elevator, throttle]. Either may be None.
    Saved to FW-FlightControl/fw_flightcontrol/data/mppi-performance/.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_dir = Path(__file__).parent.parent / 'data' / 'mppi-performance'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    labels   = ['Aileron command [-]', 'Elevator command [-]', 'Throttle command [-]']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i, (ax, ylabel) in enumerate(zip(axes, labels)):
        if mppi_actions is not None:
            t = np.arange(len(mppi_actions)) * dt
            ax.plot(t, mppi_actions[:, i], color='steelblue', linewidth=1.0, label='MPPI')
        if pid_actions is not None:
            t = np.arange(len(pid_actions)) * dt
            ax.plot(t, pid_actions[:, i], color='darkorange', linewidth=1.0, label='PID')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Action history plot saved to {save_path}")


# ============================================================================
# TEST SCRIPT UTILITIES
# ============================================================================

@contextmanager
def suppress_output():
    """Suppress all stdout/stderr including C-level file descriptors (e.g. JSBSim)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        old_fd_out = os.dup(1)
        old_fd_err = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            os.dup2(old_fd_out, 1)
            os.dup2(old_fd_err, 2)
            os.close(old_fd_out)
            os.close(old_fd_err)
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _throttle_patch(env) -> Tuple:
    """Monkey-patch apply_action so MPPI can override throttle. Returns (ref, restore_fn)."""
    from fw_jsbgym.utils import jsbsim_properties as prp
    throttle_ref = [0.3]
    original = env.unwrapped.apply_action

    def _patched(action):
        original(action)
        env.unwrapped.sim[prp.throttle_cmd] = float(np.clip(throttle_ref[0], 0.0, 1.0))

    env.unwrapped.apply_action = _patched
    return throttle_ref, lambda: setattr(env.unwrapped, 'apply_action', original)


def _safe_label(label: str) -> str:
    """Convert a run label to a filesystem-safe string."""
    return label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')


def plot_model_result(
    res: dict,
    target_roll: float,
    target_pitch: float,
    save_path: Path,
    dt: float = 0.01,
    tolerance_deg: float = 5.0,
    target_va: float = 60.0,
) -> None:
    """Save a 5-panel figure: roll, pitch, commands, angular velocities, airspeed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t = np.arange(res['steps']) * dt
    actions = res['actions']

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    ax_roll  = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[0, 1])
    ax_cmd   = fig.add_subplot(gs[1, 0])
    ax_rates = fig.add_subplot(gs[1, 1])
    ax_va    = fig.add_subplot(gs[2, :])

    ax_roll.plot(t, res['roll'], color='steelblue', linewidth=1.1, label='roll')
    ax_roll.axhline(target_roll, color='red', linestyle='--', linewidth=1.2, label='roll_ref')
    ax_roll.axhspan(target_roll - tolerance_deg, target_roll + tolerance_deg,
                    alpha=0.12, color='red')
    ax_roll.set_title('roll control')
    ax_roll.set_ylabel('roll [°]')
    ax_roll.legend(loc='upper right', fontsize=8)
    ax_roll.grid(True, alpha=0.3)

    ax_pitch.plot(t, res['pitch'], color='steelblue', linewidth=1.1, label='pitch')
    ax_pitch.axhline(target_pitch, color='red', linestyle='--', linewidth=1.2, label='pitch_ref')
    ax_pitch.axhspan(target_pitch - tolerance_deg, target_pitch + tolerance_deg,
                     alpha=0.12, color='red')
    ax_pitch.set_title('pitch control')
    ax_pitch.set_ylabel('pitch [°]')
    ax_pitch.legend(loc='upper right', fontsize=8)
    ax_pitch.grid(True, alpha=0.3)

    ax_cmd.plot(t, actions[:, 0], color='steelblue',  linewidth=0.9, label='aileron_pos_norm')
    ax_cmd.plot(t, actions[:, 1], color='darkorange', linewidth=0.9, label='elevator_pos_norm')
    ax_cmd.plot(t, actions[:, 2], color='seagreen',   linewidth=0.9, label='throttle_pos')
    ax_cmd.set_title('commands')
    ax_cmd.set_ylabel('commands [-]')
    ax_cmd.set_xlabel('time [s]')
    ax_cmd.legend(loc='upper right', fontsize=8)
    ax_cmd.grid(True, alpha=0.3)

    ax_rates.plot(t, res['p'], color='steelblue',  linewidth=0.9, label='roll_rate')
    ax_rates.plot(t, res['q'], color='darkorange', linewidth=0.9, label='pitch_rate')
    ax_rates.plot(t, res['r'], color='seagreen',   linewidth=0.9, label='yaw_rate')
    ax_rates.set_title('angular velocities')
    ax_rates.set_ylabel('angular velocities [rad/s]')
    ax_rates.set_xlabel('time [s]')
    ax_rates.legend(loc='upper right', fontsize=8)
    ax_rates.grid(True, alpha=0.3)

    ax_va.plot(t, res['va'], color='mediumpurple', linewidth=1.1, label='airspeed')
    ax_va.axhline(target_va, color='red', linestyle='--', linewidth=1.2, label=f'target {target_va:.0f} km/h')
    ax_va.set_title('airspeed')
    ax_va.set_ylabel('airspeed [km/h]')
    ax_va.set_xlabel('time [s]')
    ax_va.legend(loc='upper right', fontsize=8)
    ax_va.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def save_metrics(
    results: List[dict],
    target_roll: float,
    target_pitch: float,
    save_path: Path,
    dt: float = 0.01,
) -> None:
    """Compute per-run metrics, print a summary table, and write a CSV."""
    rows = []
    for res in results:
        rc = compute_convergence_stats(res['roll'],  target_roll,  threshold_deg=1.0)
        pc = compute_convergence_stats(res['pitch'], target_pitch, threshold_deg=1.0)

        roll_arr  = np.array(res['roll'])
        pitch_arr = np.array(res['pitch'])
        avg_t_ms  = np.mean(res['step_times']) * 1000

        rows.append({
            'label':                 res['label'],
            'steps':                 res['steps'],
            'terminated':            res['terminated'],
            'avg_step_time_ms':      f"{avg_t_ms:.2f}",
            'roll_mean_err_deg':     f"{np.mean(np.abs(roll_arr  - target_roll)):.2f}",
            'roll_converged_step':   rc['convergence_step'] if rc['convergence_step'] is not None else 'never',
            'roll_converged_s':      f"{rc['convergence_step'] * dt:.2f}" if rc['convergence_step'] is not None else 'never',
            'roll_ss_mean_err_deg':  f"{rc['steady_mean_error']:.2f}",
            'roll_ss_std_deg':       f"{rc['steady_std']:.2f}",
            'pitch_mean_err_deg':    f"{np.mean(np.abs(pitch_arr - target_pitch)):.2f}",
            'pitch_converged_step':  pc['convergence_step'] if pc['convergence_step'] is not None else 'never',
            'pitch_converged_s':     f"{pc['convergence_step'] * dt:.2f}" if pc['convergence_step'] is not None else 'never',
            'pitch_ss_mean_err_deg': f"{pc['steady_mean_error']:.2f}",
            'pitch_ss_std_deg':      f"{pc['steady_std']:.2f}",
        })

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    col_w = 30
    print(f"\n{'Model':<{col_w}} {'Steps':>6} {'Term':>5} {'t_ms':>7} "
          f"{'Roll err':>9} {'R conv':>7} {'Pitch err':>10} {'P conv':>7}")
    print('-' * (col_w + 53))
    for r in rows:
        print(f"{r['label']:<{col_w}} {r['steps']:>6} {str(r['terminated']):>5} "
              f"{r['avg_step_time_ms']:>7} {r['roll_mean_err_deg']:>9}° "
              f"{str(r['roll_converged_s']):>7} {r['pitch_mean_err_deg']:>10}° "
              f"{str(r['pitch_converged_s']):>7}")
    print(f"\nMetrics saved to {save_path}")
