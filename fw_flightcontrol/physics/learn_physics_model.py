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
import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add project to path
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import train_aphynity_epoch, HybridDynamicsODE
from fw_flightcontrol.physics.utils import (
    load_config, load_trajectory_data, compute_actual_scales, get_norm_type,
    normalize_state_torch, denormalize_state_torch,
    log_epoch_summary, log_tensorboard_epoch, save_checkpoint,
)


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

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

    residual_network.load_state_dict(checkpoint['residual_state'])
    print("  ✓ Restored network weights")

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("  ✓ Restored optimizer state")

    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        print("  ✓ Restored scheduler state")

    start_epoch    = checkpoint['epoch'] + 1
    lambda_current = checkpoint['lambda']
    train_history  = checkpoint['train_history']
    val_history    = checkpoint['val_history']

    print(f"  ✓ Resuming from epoch {start_epoch}")
    print(f"  ✓ Restored λ={lambda_current:.6f}")
    print(f"  ✓ Training history: {len(train_history['loss_total'])} epochs")
    print(f"  ✓ Validation history: {len(val_history['loss_total'])} rounds")

    return {
        'start_epoch':    start_epoch,
        'lambda_current': lambda_current,
        'train_history':  train_history,
        'val_history':    val_history,
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
    train_config    = config['training']

    # tau_1 is NOT a gradient scaling factor; it IS the Adam learning rate (APHYNITY paper)
    learning_rate = aphynity_config['tau_1']
    optimizer = torch.optim.Adam(
        residual_network.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999)
    )
    print(f"Created Adam optimizer with lr={learning_rate} (from aphynity.tau_1)")

    scheduler = None
    min_lr    = None

    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('enabled', False):
        min_lr = scheduler_config.get('min_lr', 1e-5)
        if scheduler_config.get('type') == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.7)
            )
            print(f"Created StepLR scheduler: step_size={scheduler_config.get('step_size', 20)}, "
                  f"gamma={scheduler_config.get('gamma', 0.7)}, min_lr={min_lr}")
        elif scheduler_config.get('type') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_config['num_epochs'],
                eta_min=min_lr,
            )
            print(f"Created CosineAnnealingLR scheduler: T_max={train_config['num_epochs']}, "
                  f"eta_min={min_lr}")

    return optimizer, scheduler, min_lr


# ============================================================================
# VALIDATION
# ============================================================================

def run_validation_epoch(
    hybrid_model: HybridDynamicsModel,
    val_loader,
    config: Dict,
    device: torch.device,
    lambda_current: float,
    norm_type: Optional[str],
    denorm_factors_torch: Optional[torch.Tensor],
    min_bounds_torch: Optional[torch.Tensor],
    per_state_scales_torch: Optional[torch.Tensor],
    epoch: int,
    num_epochs: int,
) -> Dict:
    """Run one full validation pass and return averaged loss metrics."""
    horizon = config['training']['horizon']

    hybrid_model.eval()
    val_metrics = {
        'loss_total':         0,
        'loss_trajectory':    0,
        'loss_regularization': 0,
        'batch_count':        0,
    }

    # Build ODE module and t_eval once — reused across all batches
    ode_module = HybridDynamicsODE(
        hybrid_model, device,
        denorm_factors=denorm_factors_torch if norm_type is not None else None,
        min_bounds=min_bounds_torch if norm_type is not None else None,
        norm_type=norm_type,
    ).to(device)
    t_eval = torch.tensor(
        [0.0, config['integration']['dt']], dtype=torch.float32, device=device
    )

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Val",
                    leave=False, unit="batch")

    with torch.no_grad():
        for batch_data in val_pbar:
            batch_initial = batch_data['initial_states'].to(device)
            batch_actions = batch_data['actions'].to(device)
            batch_states  = batch_data['states'].to(device)

            predicted_states = []
            residual_norms   = []
            current_state    = batch_initial

            for step in range(horizon):
                action = batch_actions[:, step, :]

                if norm_type is not None:
                    current_state_raw = denormalize_state_torch(current_state, denorm_factors_torch, min_bounds_torch, norm_type)
                else:
                    current_state_raw = current_state

                # arm_capture: residual norm captured from k1, no extra forward pass
                ode_module.set_action(action)
                ode_module.arm_capture()
                solution = odeint(ode_module, current_state_raw, t_eval,
                                  method=config['integration']['method'],
                                  rtol=config['integration']['rtol'],
                                  atol=config['integration']['atol'])

                if hybrid_model.with_residual and ode_module.captured_residual_norm is not None:
                    residual_norms.append(ode_module.captured_residual_norm)

                next_state_raw = solution[-1].clamp(-1000.0, 1000.0)

                if norm_type is not None:
                    next_state = normalize_state_torch(next_state_raw, denorm_factors_torch, min_bounds_torch, norm_type)
                else:
                    next_state = next_state_raw

                predicted_states.append(next_state)
                current_state = next_state

            predicted_trajectory = torch.stack(predicted_states, dim=1)

            # Loss space: normalized for data_driven, raw for bounds/no-norm
            if norm_type == 'data_driven_normalization':
                prediction_error = predicted_trajectory - batch_states
            elif norm_type is not None:
                predicted_trajectory_raw = denormalize_state_torch(predicted_trajectory, denorm_factors_torch, min_bounds_torch, norm_type)
                batch_states_raw         = denormalize_state_torch(batch_states,         denorm_factors_torch, min_bounds_torch, norm_type)
                prediction_error = predicted_trajectory_raw - batch_states_raw
            else:
                prediction_error = predicted_trajectory - batch_states

            if config['data'].get('per_state_loss_norm', False):
                prediction_error = prediction_error / (per_state_scales_torch ** 2)
            trajectory_loss    = torch.norm(prediction_error, p=2, dim=2).mean()
            regularization_loss = torch.stack(residual_norms).mean()

            batch_loss_total = regularization_loss.item() + lambda_current * trajectory_loss.item()
            val_metrics['loss_total']          += batch_loss_total
            val_metrics['loss_trajectory']     += trajectory_loss.item()
            val_metrics['loss_regularization'] += regularization_loss.item()
            val_metrics['batch_count']         += 1

            val_pbar.set_postfix({
                'L_total': f"{batch_loss_total:.4f}",
                'L_traj':  f"{trajectory_loss.item():.4f}",
            })

    for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
        val_metrics[key] /= val_metrics['batch_count']

    return val_metrics


# ============================================================================
# PRINT HELPERS (startup summaries)
# ============================================================================

def _print_training_config(
    config: Dict,
    num_epochs: int,
    batch_size: int,
    horizon: int,
    lambda_current: float,
    tau_2: float,
    lambda_min: float,
    lambda_max: float,
    denorm_factors: np.ndarray,
) -> None:
    """Print training configuration summary to stdout."""
    aphynity_config = config['aphynity']
    train_config    = config['training']

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


def _build_hyperparams_text(
    config: Dict,
    batch_size: int,
    horizon: int,
    aphynity_config: Dict,
    train_config: Dict,
) -> str:
    """Build a markdown-formatted hyperparameter summary for TensorBoard."""
    return f"""
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


# ============================================================================
# MAIN
# ============================================================================

def main(resume_checkpoint: Optional[str] = None):
    """Main training script entrypoint.

    Args:
        resume_checkpoint: Path to checkpoint to resume from. If None, starts from scratch.
    """
    print("\n" + "="*80)
    print("HYBRID PHYSICS-AUGMENTED WORLD MODEL TRAINING")
    print("="*80)
    print(f"{'Resume Mode: ' + resume_checkpoint if resume_checkpoint else 'Fresh Start Mode'}")

    config_path = Path(__file__).parent / 'training_params.yaml'
    config = load_config(str(config_path))
    print(f"\nLoaded configuration from {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    residual_network, hybrid_model = initialize_models(config, device)

    # Compile the two sub-modules that are called inside the ODE loop on every forward pass.
    # physics_prior benefits most (many fused trig/element-wise ops → fewer CUDA kernel launches).
    # residual_network also benefits (MLP forward fused into fewer ops).
    hybrid_model.physics_prior     = torch.compile(hybrid_model.physics_prior)
    hybrid_model.residual_network  = torch.compile(hybrid_model.residual_network)

    optimizer, scheduler, min_lr   = create_optimizer(residual_network, config)

    start_epoch  = 0
    resume_state = None
    if resume_checkpoint:
        resume_state = load_checkpoint(resume_checkpoint, residual_network, optimizer, scheduler, device)
        start_epoch  = resume_state['start_epoch']

    # Load data
    csv_path = Path(__file__).parent.parent / "data" / config['data']['file']
    train_loader, val_loader, _, denorm_factors, min_bounds = load_trajectory_data(str(csv_path), config)
    denorm_factors_torch = torch.tensor(denorm_factors, dtype=torch.float32, device=device)
    min_bounds_torch     = torch.tensor(min_bounds,     dtype=torch.float32, device=device)
    norm_type = get_norm_type(config)

    # Log the exact norm params that will be embedded in every checkpoint.
    # These must match what test_physics_model.py uses — copy them to config.normalization_params
    # if re-running evaluation with a different seed.
    if norm_type == 'data_driven_normalization':
        print("\n" + "="*60)
        print("NORMALIZATION PARAMS (embed these in config if evaluating later)")
        print("="*60)
        print(f"  norm_scale  (std):  {denorm_factors.tolist()}")
        print(f"  norm_offset (mean): {min_bounds.tolist()}")

    # Per-state loss scaling (optional)
    per_state_scales_torch = None
    if config['data'].get('per_state_loss_norm', False):
        print("\n  Computing actual per-state scales from training data...")
        df = pd.read_csv(str(csv_path))
        actual_scales = compute_actual_scales(df)
        per_state_scales_torch = torch.tensor(actual_scales, dtype=torch.float32, device=device)
        print(f"  Actual scales (mean |gt|): {actual_scales}")
        print(f"  Config denorm factors: {denorm_factors}")
        print(f"  Ratio (under-weighting): {denorm_factors / actual_scales}")

    # Hyperparameters
    train_config    = config['training']
    aphynity_config = config['aphynity']

    num_epochs      = train_config['num_epochs']
    horizon         = train_config['horizon']
    batch_size      = train_config['batch_size']
    log_freq        = config['logging'].get('log_freq', 10)
    val_freq        = train_config.get('val_freq', 5)
    checkpoint_freq = train_config.get('checkpoint_freq', 20)

    checkpoint_subdir   = config['logging'].get('checkpoint_subdirectory', 'checkpoints')
    checkpoint_base_dir = Path(config['logging']['checkpoint_dir']) / checkpoint_subdir

    lambda_current = resume_state['lambda_current'] if resume_state else aphynity_config['lambda_init']
    tau_2          = aphynity_config['tau_2']
    lambda_min     = aphynity_config['lambda_min']
    lambda_max     = aphynity_config['lambda_max']

    # TensorBoard
    run_name = f"{datetime.now().strftime('%y%m%d')}_{checkpoint_subdir}"
    log_base  = Path(__file__).parent.parent / "logs" / "tensorboard"
    log_dir   = log_base / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    _print_training_config(config, num_epochs, batch_size, horizon,
                           lambda_current, tau_2, lambda_min, lambda_max, denorm_factors)

    if resume_state:
        train_history = resume_state['train_history']
        val_history   = resume_state['val_history']
        print(f"\nResumed training history: {len(train_history['loss_total'])} epochs completed")
    else:
        train_history = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': [], 'lambda_history': []}
        val_history   = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': []}

    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"Training epochs {start_epoch} to {num_epochs-1} (total: {num_epochs - start_epoch})")

    global_step = 0
    config_yaml = yaml.dump(config, default_flow_style=False)

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch"):

        # Log config and hyperparams once at start of training
        if epoch == start_epoch:
            writer.add_text('Config/full_config', config_yaml)
            writer.add_text('Hyperparameters/text_summary',
                            _build_hyperparams_text(config, batch_size, horizon, aphynity_config, train_config))

        # ── Train ──────────────────────────────────────────────────────────
        hybrid_model.train()
        epoch_metrics = {k: 0 for k in [
            'loss_total', 'loss_trajectory', 'loss_regularization', 'batch_count',
            'grad_norm_before_clipping', 'grad_norm_after_clipping',
        ]}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Train",
                    leave=False, unit="batch")

        nan_batches = 0
        for batch_data in pbar:
            try:
                metrics = train_aphynity_epoch(
                    hybrid_model=hybrid_model,
                    trajectory_batch=batch_data,
                    optimizer=optimizer,
                    lambda_current=lambda_current,
                    tau_2=tau_2,
                    grad_clip_norm=train_config['grad_clip_norm'],
                    device=device,
                    ode_method=config['integration']['method'],
                    ode_rtol=config['integration']['rtol'],
                    ode_atol=config['integration']['atol'],
                    dt=config['integration']['dt'],
                    denorm_factors=denorm_factors_torch if norm_type is not None else None,
                    min_bounds=min_bounds_torch if norm_type is not None else None,
                    per_state_scales=per_state_scales_torch,
                    norm_type=norm_type,
                )
            except ValueError as e:
                nan_batches += 1
                optimizer.zero_grad()
                tqdm.write(f"  [Epoch {epoch+1}] Skipped NaN batch ({nan_batches} so far): {e}")
                continue

            epoch_metrics['loss_total']          += metrics['loss_total']
            epoch_metrics['loss_trajectory']     += metrics['loss_trajectory']
            epoch_metrics['loss_regularization'] += metrics['loss_regularization']
            epoch_metrics['batch_count']         += 1

            lambda_current = max(lambda_min, min(metrics['lambda_new'], lambda_max))

            # Per-batch TensorBoard (high-resolution)
            writer.add_scalar('Batch/loss_total',          metrics['loss_total'],          global_step)
            writer.add_scalar('Batch/loss_trajectory',     metrics['loss_trajectory'],     global_step)
            writer.add_scalar('Batch/loss_regularization', metrics['loss_regularization'], global_step)
            writer.add_scalar('Batch/lambda',              lambda_current,                 global_step)

            if 'grad_norm_before_clipping' in metrics:
                epoch_metrics['grad_norm_before_clipping'] += metrics['grad_norm_before_clipping']
                epoch_metrics['grad_norm_after_clipping']  += metrics['grad_norm_after_clipping']

            global_step += 1
            pbar.set_postfix({
                'L_total': f"{metrics['loss_total']:.4f}",
                'L_traj':  f"{metrics['loss_trajectory']:.4f}",
                'λ':       f"{lambda_current:.4f}",
            })

        # Average loss metrics over batches
        if nan_batches > 0:
            tqdm.write(f"  [Epoch {epoch+1}] Skipped {nan_batches} NaN batches out of {nan_batches + epoch_metrics['batch_count']}")
        if epoch_metrics['batch_count'] == 0:
            tqdm.write(f"  [Epoch {epoch+1}] WARNING: all batches were NaN — skipping epoch")
            continue
        for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
            epoch_metrics[key] /= epoch_metrics['batch_count']
        if epoch_metrics['batch_count'] > 0:
            epoch_metrics['grad_norm_before_clipping'] /= epoch_metrics['batch_count']
            epoch_metrics['grad_norm_after_clipping']  /= epoch_metrics['batch_count']

        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)

        # ── Validate ───────────────────────────────────────────────────────
        val_metrics = None
        if (epoch + 1) % val_freq == 0:
            val_metrics = run_validation_epoch(
                hybrid_model, val_loader, config, device, lambda_current,
                norm_type, denorm_factors_torch, min_bounds_torch, per_state_scales_torch,
                epoch, num_epochs,
            )
            val_history['loss_total'].append(val_metrics['loss_total'])
            val_history['loss_trajectory'].append(val_metrics['loss_trajectory'])
            val_history['loss_regularization'].append(val_metrics['loss_regularization'])
            hybrid_model.train()

        # ── LR Scheduler ──────────────────────────────────────────────────
        current_lr = None
        if scheduler is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr
            current_lr = optimizer.param_groups[0]['lr']

        # ── Log ────────────────────────────────────────────────────────────
        if (epoch + 1) % log_freq == 0:
            log_epoch_summary(epoch, num_epochs, epoch_metrics, val_metrics, lambda_current)

        grad_metrics = {k: epoch_metrics[k] for k in [
            'grad_norm_before_clipping', 'grad_norm_after_clipping',
        ]}
        log_tensorboard_epoch(writer, epoch, epoch_metrics, val_metrics, lambda_current, grad_metrics, current_lr)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                checkpoint_base_dir / f"epoch_{epoch+1}.pt",
                epoch, hybrid_model, optimizer, scheduler,
                lambda_current, train_history, val_history,
                norm_scale=denorm_factors if norm_type is not None else None,
                norm_offset=min_bounds    if norm_type is not None else None,
            )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final λ: {lambda_current:.4f}")
    print(f"Total epochs trained: {len(train_history['loss_total'])}")

    # Save final model (same format as epoch checkpoints so test script can read norm params)
    final_path = checkpoint_base_dir / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        final_path, num_epochs - 1, hybrid_model, optimizer, scheduler,
        lambda_current, train_history, val_history,
        norm_scale=denorm_factors if norm_type is not None else None,
        norm_offset=min_bounds    if norm_type is not None else None,
    )

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
