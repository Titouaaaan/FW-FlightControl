#!/usr/bin/env python3
"""
Training script for hybrid physics-augmented world model using APHYNITY.

This script implements the complete training pipeline:
1. Load configuration from training_params.yaml
2. Initialize physics prior and residual network
3. Load trajectory data from CSV
4. Train residual network using APHYNITY objective
5. Validate on held-out test set
6. Save checkpoints and training metrics

The key insight: combined physics-learning approach where we start with a
physics prior (which captures the known aerodynamic structure) and learn
a residual network to correct systematic errors. This is more data-efficient
and generalizes better than learning from scratch.
"""

import torch
import torch.nn as nn
import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add project to path
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import train_aphynity_epoch, HybridDynamicsODE


# ============================================================================
# NORMALIZATION UTILITIES
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


def denormalize_derivative(derivative: np.ndarray, denorm_factors: np.ndarray) -> np.ndarray:
    """
    Denormalize derivative from normalized space to raw space.
    
    Converts ds_norm/dt (change in normalized state) to ds_raw/dt.
    
    Args:
        derivative: normalized derivative, shape (..., state_dim)
        denorm_factors: (max-min)/2 for each state, shape (state_dim,)
    
    Returns:
        raw derivative in same units as raw state changes
    """
    return derivative * denorm_factors


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trajectory_data(csv_path: str, config: Dict) -> Tuple:
    """Load trajectory data from CSV and prepare training/validation/test sets."""
    
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    print(f"\nLoading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transitions from {df['trajectory_id'].nunique()} trajectories")
    
    # Configuration parameters
    horizon = config['training']['horizon']
    state_dim = config['network']['state_dim']
    action_dim = config['network']['action_dim']
    normalize = config['data']['normalize']
    batch_size = config['training']['batch_size']
    
    # Define column names: State indices [0-5] + [8-9] (skip [6-7] which are control errors)
    # CSV has 14 states: [0-5]=kinematics, [6-7]=errors, [8-9]=aerodynamic angles, [10-13]=commands/integrals
    state_indices = [0, 1, 2, 3, 4, 5, 8, 9]
    state_cols = [f's_t_{i}' for i in state_indices]
    action_cols = [f'a_t_{i}' for i in range(action_dim)]
    next_state_cols = [f's_t+1_{i}' for i in state_indices]
    
    # Extract bounds for normalization (from config)
    min_bounds, max_bounds = extract_bounds_from_config(config)
    denorm_factors = compute_denorm_factors(min_bounds, max_bounds)
    
    print(f"\nState bounds loaded from config:")
    print(f"  Min: {min_bounds}")
    print(f"  Max: {max_bounds}")
    print(f"  Denorm factors: {denorm_factors}")
    
    
    # Extract sequences from each trajectory
    trajectory_sequences_by_traj = {}
    
    for traj_id, group in df.groupby('trajectory_id'):
        group = group.sort_values('step_id').reset_index(drop=True)
        
        # Extract numpy arrays for this trajectory
        states = group[state_cols].values.copy()
        actions = group[action_cols].values
        next_states = group[next_state_cols].values.copy()
        
        # Apply unit conversions: airspeed from km/h to m/s (index 2 in state vector)
        states[:, 2] = states[:, 2] / 3.6
        next_states[:, 2] = next_states[:, 2] / 3.6
        
        num_steps = len(states)
        traj_sequences = []
        
        for start_idx in range(num_steps - horizon):
            seq_states = states[start_idx:start_idx + horizon]
            seq_actions = actions[start_idx:start_idx + horizon]
            seq_next_states = next_states[start_idx:start_idx + horizon]
            
            if normalize:
                seq_states = normalize_state(seq_states, min_bounds, max_bounds)
                seq_next_states = normalize_state(seq_next_states, min_bounds, max_bounds)
            
            traj_sequences.append({
                'initial_state': seq_states[0].copy(),
                'actions': seq_actions.copy(),
                'states': seq_next_states.copy(),
            })
        
        trajectory_sequences_by_traj[traj_id] = traj_sequences
    
    trajectories = sorted(list(df['trajectory_id'].unique()))
    random_seed = config['data'].get('random_seed', 42)
    np.random.seed(random_seed)
    np.random.shuffle(trajectories)
    
    n_traj_train = int(len(trajectories) * config['data']['train_fraction'])
    n_traj_val = int(len(trajectories) * config['data']['val_fraction'])
    
    train_traj_ids = set(trajectories[:n_traj_train])
    val_traj_ids = set(trajectories[n_traj_train:n_traj_train + n_traj_val])
    test_traj_ids = set(trajectories[n_traj_train + n_traj_val:])
    
    # Split sequences by trajectory membership
    train_seqs = [seq for traj_id, seq_list in trajectory_sequences_by_traj.items() 
                  if traj_id in train_traj_ids for seq in seq_list]
    val_seqs = [seq for traj_id, seq_list in trajectory_sequences_by_traj.items() 
                if traj_id in val_traj_ids for seq in seq_list]
    test_seqs = [seq for traj_id, seq_list in trajectory_sequences_by_traj.items() 
                 if traj_id in test_traj_ids for seq in seq_list]
    
    print(f"\nTrain/Val/Test split (at trajectory level):")
    print(f"  Trajectories: {len(train_traj_ids)} train, {len(val_traj_ids)} val, {len(test_traj_ids)} test")
    print(f"  Sequences: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test")
    
    # Create dataset class that batches sequences into tensors
    class TrajectoryDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            seq = self.sequences[idx]
            return {
                'initial_states': torch.tensor(seq['initial_state'], dtype=torch.float32),
                'actions': torch.tensor(seq['actions'], dtype=torch.float32),
                'states': torch.tensor(seq['states'], dtype=torch.float32),
            }
        
        def _collate_fn(batch):
            """Stack batch sequences into proper shapes for training."""
            initial_states = torch.stack([b['initial_states'] for b in batch])  # (B, 8)
            actions = torch.stack([b['actions'] for b in batch])  # (B, H, 3)
            states = torch.stack([b['states'] for b in batch])  # (B, H, 8)
            return {
                'initial_states': initial_states,
                'actions': actions,
                'states': states,
            }
    
    train_dataset = TrajectoryDataset(train_seqs)
    val_dataset = TrajectoryDataset(val_seqs)
    test_dataset = TrajectoryDataset(test_seqs)
    
    # Create dataloaders with proper collation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=TrajectoryDataset._collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TrajectoryDataset._collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TrajectoryDataset._collate_fn
    )
    
    return train_loader, val_loader, test_loader, denorm_factors, min_bounds


def initialize_models(config: Dict, device: torch.device) -> Tuple[PhysicsAugmented, HybridDynamicsModel]:
    """
    Initialize physics prior and residual network.
    
    The physics prior is loaded from pre-computed aerodynamic coefficients
    and is never trained (frozen). The residual network is trained to learn
    corrections to the physics prior's predictions.
    """
    print("\n" + "="*80)
    print("INITIALIZING MODELS")
    print("="*80)
    
    # Physics prior (frozen, not trained)
    print("\nPhysics Prior:")
    physics_prior = PhysicsPrior()
    print("  ✓ Loaded aerodynamic coefficients from aero_coefficients.yaml")
    print("  ✓ Model is frozen (non-trainable)")
    
    # Residual network (trainable)
    print("\nResidual Network:")
    net_config = config['network']
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=net_config['hidden_dims'],
        activation=net_config['activation'],
        use_batch_norm=net_config['use_batch_norm']
    )
    num_residual_params = sum(p.numel() for p in residual_network.parameters())
    print(f"  ✓ Created MLP with {num_residual_params:,} trainable parameters")
    
    # Print architecture
    input_size = net_config['state_dim'] + net_config['action_dim']
    architecture_str = f"{input_size} -> " + " -> ".join(map(str, net_config['hidden_dims'])) + f" -> {net_config['state_dim']}"
    print(f"  ✓ Architecture: {architecture_str}")
    
    # Hybrid model combines both
    print("\nHybrid Dynamics Model:")
    integration_method = config.get('integration', {}).get('method', 'rk4')
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True,
        integration_method=integration_method
    )
    hybrid_model = hybrid_model.to(device)
    print(f"  ✓ Initialized (ds/dt = F_p + F_a)")
    print(f"  ✓ Integration method: {integration_method}")
    print(f"  ✓ Model moved to {device}")
    
    return residual_network, hybrid_model


def load_checkpoint(checkpoint_path: str, residual_network: PhysicsAugmented, optimizer, scheduler, device: torch.device) -> Dict:
    """
    Load a checkpoint and restore training state.
    
    Returns:
        Dictionary with restored state: epoch, lambda_current, train_history, val_history
    """
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore network weights
    residual_network.load_state_dict(checkpoint['residual_state'])
    print("  ✓ Restored network weights")
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("  ✓ Restored optimizer state")
    
    # Restore scheduler state if present
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        print("  ✓ Restored scheduler state")
    
    # Restore training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    lambda_current = checkpoint['lambda']
    train_history = checkpoint['train_history']
    val_history = checkpoint['val_history']
    
    print(f"  ✓ Resuming from epoch {start_epoch}")
    print(f"  ✓ Restored λ={lambda_current:.6f}")
    print(f"  ✓ Training history: {len(train_history['loss_total'])} epochs")
    print(f"  ✓ Validation history: {len(val_history['loss_total'])} rounds")
    
    return {
        'start_epoch': start_epoch,
        'lambda_current': lambda_current,
        'train_history': train_history,
        'val_history': val_history,
    }


def create_optimizer(residual_network: PhysicsAugmented, config: Dict):
    """
    Create optimizer for residual network parameters.
    
    We only optimize the residual network weights. The physics prior
    is frozen and provides fixed baseline predictions.
    
    Returns:
        tuple: (optimizer, scheduler, min_lr) if scheduler enabled
               (optimizer, None, None) if scheduler disabled
    """
    aphynity_config = config['aphynity']
    train_config = config['training']
    
    # Adam optimizer with tau_1 as learning rate (official APHYNITY approach)
    # tau_1 is NOT a gradient scaling factor, it's the learning rate for Adam
    learning_rate = aphynity_config['tau_1']
    optimizer = torch.optim.Adam(
        residual_network.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999)  # Standard betas from official APHYNITY
    )
    
    print(f"Created Adam optimizer with lr={learning_rate} (from aphynity.tau_1)")
    
    # Create scheduler if enabled in config
    scheduler = None
    min_lr = None
    
    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('enabled', False):
        if scheduler_config.get('type') == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.7)
            )
            min_lr = scheduler_config.get('min_lr', 1e-5)
            print(f"Created StepLR scheduler: step_size={scheduler_config.get('step_size', 20)}, "
                  f"gamma={scheduler_config.get('gamma', 0.7)}, min_lr={min_lr}")
    
    return optimizer, scheduler, min_lr


def main(resume_checkpoint: Optional[str] = None):
    """Main training script entrypoint.
    
    Args:
        resume_checkpoint: Path to checkpoint to resume from. If None, starts from scratch.
    """
    print("\n" + "="*80)
    print("HYBRID PHYSICS-AUGMENTED WORLD MODEL TRAINING")
    print("="*80)
    
    if resume_checkpoint:
        print(f"Resume Mode: {resume_checkpoint}")
    else:
        print("Fresh Start Mode")
    
    # Load configuration from the physics directory
    config_path = Path(__file__).parent / 'training_params.yaml'
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(str(config_path))
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    residual_network, hybrid_model = initialize_models(config, device)
    
    # Create optimizer (only for residual network; physics prior is frozen)
    optimizer, scheduler, min_lr = create_optimizer(residual_network, config)
    
    # Load checkpoint if resuming
    start_epoch = 0
    resume_state = None
    if resume_checkpoint:
        resume_state = load_checkpoint(resume_checkpoint, residual_network, optimizer, scheduler, device)
        start_epoch = resume_state['start_epoch']
    
    # Load trajectory data from CSV
    csv_path = Path(__file__).parent.parent / "data" / "updated_trajectory_data_progressive_noatmo.csv"
    train_loader, val_loader, test_loader, denorm_factors, min_bounds = load_trajectory_data(str(csv_path), config)
    denorm_factors_torch = torch.tensor(denorm_factors, dtype=torch.float32, device=device)
    min_bounds_torch = torch.tensor(min_bounds, dtype=torch.float32, device=device)
    
    # Compute actual scales from data for per-state loss scaling (if enabled)
    # These reflect what's actually in the data, not config bounds
    use_actual_scales = config['data'].get('per_state_loss_norm', False)
    if use_actual_scales:
        print("\n  Computing actual per-state scales from training data...")
        df = pd.read_csv(str(csv_path))
        actual_scales = compute_actual_scales(df)
        per_state_scales_torch = torch.tensor(actual_scales, dtype=torch.float32, device=device)
        print(f"  Actual scales (mean |gt|): {actual_scales}")
        print(f"  Config denorm factors: {denorm_factors}")
        print(f"  Ratio (under-weighting): {denorm_factors / actual_scales}")
    else:
        # If not using per-state scaling, still pass denorm_factors for denorm/renorm in ODE
        per_state_scales_torch = None
    
    # Extract training hyperparameters
    train_config = config['training']
    aphynity_config = config['aphynity']
    
    num_epochs = train_config['num_epochs']
    horizon = train_config['horizon']
    batch_size = train_config['batch_size']
    log_freq = config['logging'].get('log_freq', 10)
    val_freq = train_config.get('val_freq', 5)
    checkpoint_freq = train_config.get('checkpoint_freq', 20)
    
    # Construct checkpoint directory path
    checkpoint_subdir = config['logging'].get('checkpoint_subdirectory', 'checkpoints')
    checkpoint_base_dir = Path(config['logging']['checkpoint_dir']) / checkpoint_subdir
    print(f"\nCheckpoint directory: {checkpoint_base_dir}")
    
    # Use resumed state if available, otherwise use config defaults
    if resume_state:
        lambda_current = resume_state['lambda_current']
    else:
        lambda_current = aphynity_config['lambda_init']
    
    tau_2 = aphynity_config['tau_2']
    lambda_min = aphynity_config['lambda_min']
    lambda_max = aphynity_config['lambda_max']
    
    # Initialize TensorBoard writer
    # If resuming, append to existing logs; otherwise create new timestamped dir
    if resume_state:
        # During resume, we'll continue writing to existing log directory
        # (or create a new one - for safety, we create new to separate runs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_base = Path(__file__).parent.parent / "logs" / "tensorboard"
        log_dir = log_base / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nNew TensorBoard log directory: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_base = Path(__file__).parent.parent / "logs" / "tensorboard"
        log_dir = log_base / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    
    print("\nTraining Loop:")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Learning rate: {aphynity_config['tau_1']}")
    print(f"  • Grad clipping: {train_config['grad_clip_norm']}")
    
    print("\nAPHYNITY Objective:")
    print(f"  • Prediction horizon: {horizon} steps (~{horizon*config['integration']['dt']:.2f}s)")
    print(f"  • Initial λ: {lambda_current}")
    print(f"  • λ step size (τ_2): {tau_2}")
    print(f"  • λ bounds: [{lambda_min}, {lambda_max}]")

    if config['data'].get('per_state_loss_norm', False):
        print("\nPer-state loss normalization (half-range scales):")
        print(f"  • Scales: {denorm_factors}")
    
    if resume_state:
        train_history = resume_state['train_history']
        val_history = resume_state['val_history']
        print(f"\nResumed training history: {len(train_history['loss_total'])} epochs completed")
    else:
        train_history = {
            'loss_total': [],
            'loss_trajectory': [],
            'loss_regularization': [],
            'lambda_history': [],
        }
        val_history = {
            'loss_total': [],
            'loss_trajectory': [],
            'loss_regularization': [],
        }
    
    # ========================================================================
    # TRAINING LOOP - Following the APHYNITY pseudocode
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"Training epochs {start_epoch} to {num_epochs-1} (total: {num_epochs - start_epoch})")
    
    # Global step counter for per-batch TensorBoard logging
    # Offset by number of already-completed epochs if resuming
    global_step = 0
    
    # Log configuration to TensorBoard only on first epoch (not on resume)
    import yaml
    config_yaml = yaml.dump(config, default_flow_style=False)
    
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch"):
        # Log config on first epoch
        if epoch == 0:
            writer.add_text('Config/full_config', config_yaml)
            
            # Format hyperparameters as readable text (not scalars/plots)
            hyperparams_text = f"""
## Training Hyperparameters

**Learning:**
- Learning Rate: {aphynity_config['tau_1']} (from aphynity.tau_1)
- Optimizer: Adam
- Gradient Clip Norm: {train_config.get('grad_clip_norm', 1.0)}

**Scheduling:**
- Scheduler Enabled: {train_config.get('scheduler', {}).get('enabled', False)}
- Scheduler Type: {train_config.get('scheduler', {}).get('type', 'none')}
- Step Size: {train_config.get('scheduler', {}).get('step_size', 'N/A')}
- Gamma: {train_config.get('scheduler', {}).get('gamma', 'N/A')}
- Min LR: {train_config.get('scheduler', {}).get('min_lr', 'N/A')}

**Data & Architecture:**
- Batch Size: {batch_size}
- Horizon: {horizon} steps (~{horizon*config['integration']['dt']:.2f}s)
- State Dim: {config['network']['state_dim']}
- Action Dim: {config['network']['action_dim']}
- Hidden Dims: {config['network']['hidden_dims']}

**APHYNITY:**
- Lambda Init: {aphynity_config['lambda_init']}
- Tau 1: {aphynity_config['tau_1']}
- Tau 2: {aphynity_config['tau_2']}
- Lambda Bounds: [{aphynity_config['lambda_min']}, {aphynity_config['lambda_max']}]

**Integration:**
- ODE Method: {config['integration']['method']}
- rtol: {config['integration']['rtol']}
- atol: {config['integration']['atol']}
"""
            writer.add_text('Hyperparameters/text_summary', hyperparams_text)
        # ====================================================================
        # Training epoch: process all batches
        # ====================================================================
        hybrid_model.train()
        epoch_metrics = {
            'loss_total': 0,
            'loss_trajectory': 0,
            'loss_regularization': 0,
            'batch_count': 0,
            'grad_norm_before_clipping': 0,
            'grad_norm_after_clipping': 0,
            'grad_max_before_clipping': 0,
            'grad_max_after_clipping': 0,
            'grad_norm_clipped': 0,
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Train", 
                    leave=False, unit="batch")
        
        for batch_data in pbar:
            # Train one batch using APHYNITY objective
            metrics = train_aphynity_epoch(
                hybrid_model=hybrid_model,
                trajectory_batch=batch_data,
                optimizer=optimizer,
                lambda_current=lambda_current,
                tau_1=config['aphynity']['tau_1'],
                tau_2=tau_2,
                device=device,
                ode_method=config['integration']['method'],
                ode_rtol=config['integration']['rtol'],
                ode_atol=config['integration']['atol'],
                dt=config['integration']['dt'],
                denorm_factors=denorm_factors_torch if config['data']['normalize'] else None,
                min_bounds=min_bounds_torch if config['data']['normalize'] else None,
                per_state_scales=per_state_scales_torch  # Use actual scales when per_state_loss_norm enabled
            )
            
            # Accumulate metrics for epoch-level statistics
            epoch_metrics['loss_total'] += metrics['loss_total']
            epoch_metrics['loss_trajectory'] += metrics['loss_trajectory']
            epoch_metrics['loss_regularization'] += metrics['loss_regularization']
            epoch_metrics['batch_count'] += 1
            
            # Update lambda for next batch (dual ascent)
            lambda_current = metrics['lambda_new']
            lambda_current = max(lambda_min, min(lambda_current, lambda_max))
            
            # Log per-batch metrics to TensorBoard (high-resolution data)
            writer.add_scalar('Batch/loss_total', metrics['loss_total'], global_step)
            writer.add_scalar('Batch/loss_trajectory', metrics['loss_trajectory'], global_step)
            writer.add_scalar('Batch/loss_regularization', metrics['loss_regularization'], global_step)
            writer.add_scalar('Batch/lambda', lambda_current, global_step)
            
            # Accumulate gradient statistics
            if 'grad_norm_before_clipping' in metrics:
                epoch_metrics['grad_norm_before_clipping'] = epoch_metrics.get('grad_norm_before_clipping', 0) + metrics['grad_norm_before_clipping']
                epoch_metrics['grad_max_before_clipping'] = max(epoch_metrics.get('grad_max_before_clipping', 0), metrics['grad_max_before_clipping'])
                epoch_metrics['grad_norm_after_clipping'] = epoch_metrics.get('grad_norm_after_clipping', 0) + metrics['grad_norm_after_clipping']
                epoch_metrics['grad_max_after_clipping'] = max(epoch_metrics.get('grad_max_after_clipping', 0), metrics['grad_max_after_clipping'])
                epoch_metrics['grad_norm_clipped'] = max(epoch_metrics.get('grad_norm_clipped', 0), metrics['grad_norm_clipped'])
            
            global_step += 1
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'L_total': f"{metrics['loss_total']:.4f}",
                'L_traj': f"{metrics['loss_trajectory']:.4f}",
                'λ': f"{lambda_current:.4f}"
            })
        
        # Average metrics over batches and compute gradient stats
        for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
            epoch_metrics[key] /= epoch_metrics['batch_count']
        
        # Average gradient metrics if they were collected
        if 'grad_norm_before_clipping' in epoch_metrics:
            epoch_metrics['grad_norm_before_clipping'] /= epoch_metrics['batch_count']
            epoch_metrics['grad_norm_after_clipping'] /= epoch_metrics['batch_count']
        
        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)
        
        # Log epoch-level metrics to TensorBoard (aggregated statistics)
        writer.add_scalar('Epoch/train_loss_total', epoch_metrics['loss_total'], epoch)
        writer.add_scalar('Epoch/train_loss_trajectory', epoch_metrics['loss_trajectory'], epoch)
        writer.add_scalar('Epoch/train_loss_regularization', epoch_metrics['loss_regularization'], epoch)
        writer.add_scalar('Epoch/lambda_final', lambda_current, epoch)
        
        # Log gradient statistics per epoch
        if 'grad_norm_before_clipping' in epoch_metrics:
            writer.add_scalar('Gradients/norm_before_clipping', epoch_metrics['grad_norm_before_clipping'], epoch)
            writer.add_scalar('Gradients/norm_after_clipping', epoch_metrics['grad_norm_after_clipping'], epoch)
            writer.add_scalar('Gradients/max_norm_before_clipping', epoch_metrics['grad_max_before_clipping'], epoch)
            writer.add_scalar('Gradients/max_norm_after_clipping', epoch_metrics['grad_max_after_clipping'], epoch)
            writer.add_scalar('Gradients/clipping_threshold', epoch_metrics['grad_norm_clipped'], epoch)
        
        # Step learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step()
            # Apply minimum learning rate constraint
            for param_group in optimizer.param_groups:
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr
            # Log current learning rate to TensorBoard
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/learning_rate', current_lr, epoch)
        
        # ====================================================================
        # Validation: assess performance on held-out set
        # ====================================================================
        if (epoch + 1) % val_freq == 0:
            hybrid_model.eval()
            val_metrics = {
                'loss_total': 0,
                'loss_trajectory': 0,
                'loss_regularization': 0,
                'batch_count': 0,
            }
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Val", 
                           leave=False, unit="batch")
            
            with torch.no_grad():
                for batch_data in val_pbar:
                    # Compute losses without updating parameters
                    batch_initial = batch_data['initial_states'].to(device)
                    batch_actions = batch_data['actions'].to(device)
                    batch_states = batch_data['states'].to(device)
                    
                    # Unroll trajectory using proper ODE wrapper (same as training)
                    from torchdiffeq import odeint
                    predicted_states = []
                    residual_norms = []
                    current_state = batch_initial
                    
                    # Create ODE wrapper once per batch
                    ode_module = HybridDynamicsODE(hybrid_model, device, 
                                                  denorm_factors=denorm_factors_torch if config['data']['normalize'] else None,
                                                  min_bounds=min_bounds_torch if config['data']['normalize'] else None).to(device)
                    
                    for step in range(horizon):
                        action = batch_actions[:, step, :]
                        residual_output = hybrid_model.residual_network(current_state, action)
                        # Denormalize residual to raw space before penalizing (if using normalization)
                        # This ensures validation regularization loss matches training computation
                        if config['data']['normalize']:
                            residual_output_raw = residual_output * denorm_factors_torch
                            residual_norm = torch.norm(residual_output_raw, p=2, dim=1).mean()
                        else:
                            residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
                        residual_norms.append(residual_norm)
                        
                        # Denormalize current state to raw space for ODE integration
                        if config['data']['normalize']:
                            current_state_raw = (current_state + 1.0) * denorm_factors_torch + min_bounds_torch
                        else:
                            current_state_raw = current_state
                        
                        # Set action and integrate in raw space
                        ode_module.set_action(action)
                        t_eval = torch.tensor([0.0, config['integration']['dt']], dtype=current_state_raw.dtype, device=device)
                        solution = odeint(ode_module, current_state_raw, t_eval, 
                                        method=config['integration']['method'],
                                        rtol=config['integration']['rtol'],
                                        atol=config['integration']['atol'])
                        next_state_raw = solution[-1]
                        # Clamp to valid physical bounds
                        next_state_raw = next_state_raw.clamp(-1000.0, 1000.0)
                        
                        # Renormalize to normalized space
                        if config['data']['normalize']:
                            next_state = 2.0 * (next_state_raw - min_bounds_torch) / (2.0 * denorm_factors_torch) - 1.0
                        else:
                            next_state = next_state_raw
                        
                        predicted_states.append(next_state)
                        current_state = next_state
                    
                    predicted_trajectory = torch.stack(predicted_states, dim=1)
                    
                    # Denormalize predictions and ground truth to raw space (same as training)
                    # This ensures validation loss is computed in raw physical space
                    if config['data']['normalize']:
                        predicted_trajectory_raw = (predicted_trajectory + 1.0) * denorm_factors_torch + min_bounds_torch
                        batch_states_raw = (batch_states + 1.0) * denorm_factors_torch + min_bounds_torch
                    else:
                        predicted_trajectory_raw = predicted_trajectory
                        batch_states_raw = batch_states
                    
                    prediction_error = predicted_trajectory_raw - batch_states_raw
                    if config['data'].get('per_state_loss_norm', False):
                        prediction_error = prediction_error / (per_state_scales_torch ** 2)  # Use actual scales with inverse scaling
                    trajectory_loss = torch.norm(prediction_error, p=2, dim=2).mean()
                    regularization_loss = torch.stack(residual_norms).mean()
                    
                    # APHYNITY loss (same as training): regularization + λ * trajectory
                    # Note: τ_1 is gradient scaling applied during training, not part of loss value
                    batch_loss_total = regularization_loss.item() + lambda_current * trajectory_loss.item()
                    val_metrics['loss_total'] += batch_loss_total
                    val_metrics['loss_trajectory'] += trajectory_loss.item()
                    val_metrics['loss_regularization'] += regularization_loss.item()
                    val_metrics['batch_count'] += 1
                    # Update progress bar with current batch metrics
                    val_pbar.set_postfix({
                        'L_total': f"{batch_loss_total:.4f}",
                        'L_traj': f"{trajectory_loss.item():.4f}",
                    })
            
            for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
                val_metrics[key] /= val_metrics['batch_count']
            
            val_history['loss_total'].append(val_metrics['loss_total'])
            val_history['loss_trajectory'].append(val_metrics['loss_trajectory'])
            val_history['loss_regularization'].append(val_metrics['loss_regularization'])
            
            # Log epoch-level validation metrics to TensorBoard
            writer.add_scalar('Epoch/val_loss_total', val_metrics['loss_total'], epoch)
            writer.add_scalar('Epoch/val_loss_trajectory', val_metrics['loss_trajectory'], epoch)
            writer.add_scalar('Epoch/val_loss_regularization', val_metrics['loss_regularization'], epoch)
        
        # ====================================================================
        # Logging
        # ====================================================================
        if (epoch + 1) % log_freq == 0:
            log_str = f"Epoch {epoch+1:3d}/{num_epochs} | "
            log_str += f"L_total={epoch_metrics['loss_total']:.4f} | "
            log_str += f"L_traj={epoch_metrics['loss_trajectory']:.4f} | "
            log_str += f"L_reg={epoch_metrics['loss_regularization']:.4f} | "
            log_str += f"λ={lambda_current:.4f}"
            
            if (epoch + 1) % val_freq == 0:
                log_str += f" | Val_L={val_metrics['loss_total']:.4f}"
            
            print(log_str)
        
        # ====================================================================
        # Checkpointing
        # ====================================================================
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = checkpoint_base_dir / f"epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_dict = {
                'epoch': epoch,
                'residual_state': hybrid_model.residual_network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lambda': lambda_current,
                'train_history': train_history,
                'val_history': val_history,
            }
            
            # Include scheduler state if active
            if scheduler is not None:
                checkpoint_dict['scheduler_state'] = scheduler.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final λ: {lambda_current:.4f}")
    print(f"Total epochs trained: {len(train_history['loss_total'])}")
    
    # Save final model
    final_path = checkpoint_base_dir / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hybrid_model.residual_network.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"  → Run: tensorboard --logdir {log_base}")
    print("  → Then open http://localhost:6006 in your browser")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid physics-augmented world model')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (e.g., fw_flightcontrol/physics/checkpoints/epoch_100.pt)'
    )
    args = parser.parse_args()
    
    main(resume_checkpoint=args.resume)
