#!/usr/bin/env python3
"""Training script for hybrid physics-augmented world model using APHYNITY."""

import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.set_float32_matmul_precision('high')

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import train_aphynity_batch, HybridDynamicsODE, rollout
from fw_flightcontrol.physics.utils import (
    load_config, load_trajectory_data, get_norm_type,
    log_epoch_summary, log_tensorboard_epoch, save_checkpoint,
    clean_state_dict_for_compilation,
)


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_models(config: Dict, device: torch.device):
    physics_prior    = PhysicsPrior()
    net_config       = config['network']
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=net_config['hidden_dims'],
        activation=net_config['activation'],
        prev_action_dim=net_config.get('prev_action_dim', 0),
    )
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
    ).to(device)
    return residual_network, hybrid_model


def load_checkpoint(checkpoint_path: str, residual_network: PhysicsAugmented,
                    optimizer, scheduler, device: torch.device) -> Dict:
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = clean_state_dict_for_compilation(checkpoint['residual_state'])
    residual_network.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    start_epoch    = checkpoint['epoch'] + 1
    lambda_current = checkpoint['lambda']
    train_history  = checkpoint['train_history']
    val_history    = checkpoint['val_history']
    print(f"  Resuming from epoch {start_epoch}, λ={lambda_current:.6f}")
    return {
        'start_epoch':    start_epoch,
        'lambda_current': lambda_current,
        'train_history':  train_history,
        'val_history':    val_history,
    }


def create_optimizer(residual_network: PhysicsAugmented, config: Dict):
    aphynity_config  = config['aphynity']
    train_config     = config['training']
    learning_rate    = aphynity_config['tau_1']
    optimizer = torch.optim.Adam(residual_network.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    scheduler = None
    min_lr    = None
    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('enabled', False):
        min_lr = scheduler_config.get('min_lr', 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config['num_epochs'], eta_min=min_lr,
        )

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
    epoch: int,
    num_epochs: int,
) -> Dict:
    """Run one full validation pass and return averaged loss metrics."""
    import torch.nn as nn

    hybrid_model.eval()
    val_metrics = {'loss_total': 0, 'loss_trajectory': 0, 'loss_regularization': 0, 'batch_count': 0}

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
            batch_initial      = batch_data['initial_states'].to(device)
            batch_actions      = batch_data['actions'].to(device)
            batch_states       = batch_data['states'].to(device)
            batch_prev_actions = batch_data.get('prev_actions')
            if batch_prev_actions is not None:
                batch_prev_actions = batch_prev_actions.to(device)

            predicted_trajectory = rollout(
                ode_module, batch_initial, batch_actions, t_eval,
                ode_method=config['integration']['method'],
                ode_rtol=config['integration']['rtol'],
                ode_atol=config['integration']['atol'],
                denorm_factors=denorm_factors_torch if norm_type is not None else None,
                min_bounds=min_bounds_torch if norm_type is not None else None,
                norm_type=norm_type,
                prev_actions=batch_prev_actions,
            )

            trajectory_loss = nn.functional.mse_loss(predicted_trajectory, batch_states)

            if hybrid_model.with_residual:
                residual_reg = hybrid_model.residual_network(
                    batch_initial, batch_actions[:, 0, :],
                    batch_prev_actions[:, 0, :] if batch_prev_actions is not None else None,
                )
                regularization = ((residual_reg * denorm_factors_torch) ** 2).mean() if denorm_factors_torch is not None \
                                 else (residual_reg ** 2).mean()
            else:
                regularization = torch.tensor(0.0, device=device)

            batch_loss_total = regularization.item() + lambda_current * trajectory_loss.item()
            val_metrics['loss_total']          += batch_loss_total
            val_metrics['loss_trajectory']     += trajectory_loss.item()
            val_metrics['loss_regularization'] += regularization.item()
            val_metrics['batch_count']         += 1

            val_pbar.set_postfix({
                'L_total': f"{batch_loss_total:.4f}",
                'L_traj':  f"{trajectory_loss.item():.4f}",
            })

    for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
        val_metrics[key] /= val_metrics['batch_count']

    return val_metrics


# ============================================================================
# MAIN
# ============================================================================

def main(resume_checkpoint: Optional[str] = None):
    config_path = Path(__file__).parent / 'training_params.yaml'
    config      = load_config(str(config_path))
    net_config  = config['network']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    residual_network, hybrid_model = initialize_models(config, device)

    hybrid_model.physics_prior    = torch.compile(hybrid_model.physics_prior)
    hybrid_model.residual_network = torch.compile(hybrid_model.residual_network)

    optimizer, scheduler, min_lr = create_optimizer(residual_network, config)

    start_epoch  = 0
    resume_state = None
    if resume_checkpoint:
        resume_state = load_checkpoint(resume_checkpoint, residual_network, optimizer, scheduler, device)
        start_epoch  = resume_state['start_epoch']

    csv_path = Path(__file__).parent.parent / "data" / config['data']['file']
    train_loader, val_loader, _, denorm_factors, min_bounds = load_trajectory_data(str(csv_path), config)
    denorm_factors_torch = torch.tensor(denorm_factors, dtype=torch.float32, device=device)
    min_bounds_torch     = torch.tensor(min_bounds,     dtype=torch.float32, device=device)
    norm_type = get_norm_type(config)

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

    run_name = f"{datetime.now().strftime('%y%m%d')}_{checkpoint_subdir}"
    log_base  = Path(__file__).parent.parent / "logs" / "tensorboard"
    log_dir   = log_base / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    if resume_state:
        train_history = resume_state['train_history']
        val_history   = resume_state['val_history']
    else:
        train_history = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': [], 'lambda_history': []}
        val_history   = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': []}

    print(f"Training epochs {start_epoch} to {num_epochs-1} | batch={batch_size} | H={horizon}")

    global_step = 0

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch"):

        hybrid_model.train()
        epoch_metrics = {k: 0 for k in [
            'loss_total', 'loss_trajectory', 'loss_regularization', 'batch_count', 'grad_norm',
        ]}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Train",
                    leave=False, unit="batch")

        nan_batches = 0
        for batch_data in pbar:
            try:
                metrics = train_aphynity_batch(
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
            epoch_metrics['grad_norm']           += metrics['grad_norm']
            epoch_metrics['batch_count']         += 1

            lambda_current = max(lambda_min, min(metrics['lambda_new'], lambda_max))

            writer.add_scalar('Batch/loss_total',          metrics['loss_total'],          global_step)
            writer.add_scalar('Batch/loss_trajectory',     metrics['loss_trajectory'],     global_step)
            writer.add_scalar('Batch/loss_regularization', metrics['loss_regularization'], global_step)
            writer.add_scalar('Batch/lambda',              lambda_current,                 global_step)

            global_step += 1
            pbar.set_postfix({
                'L_total': f"{metrics['loss_total']:.4f}",
                'L_traj':  f"{metrics['loss_trajectory']:.4f}",
                'λ':       f"{lambda_current:.4f}",
            })

        if nan_batches > 0:
            tqdm.write(f"  [Epoch {epoch+1}] Skipped {nan_batches} NaN batches")
        if epoch_metrics['batch_count'] == 0:
            tqdm.write(f"  [Epoch {epoch+1}] WARNING: all batches were NaN — skipping epoch")
            continue
        for key in ['loss_total', 'loss_trajectory', 'loss_regularization', 'grad_norm']:
            epoch_metrics[key] /= epoch_metrics['batch_count']

        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)

        val_metrics = None
        if (epoch + 1) % val_freq == 0:
            val_metrics = run_validation_epoch(
                hybrid_model, val_loader, config, device, lambda_current,
                norm_type, denorm_factors_torch, min_bounds_torch,
                epoch, num_epochs,
            )
            val_history['loss_total'].append(val_metrics['loss_total'])
            val_history['loss_trajectory'].append(val_metrics['loss_trajectory'])
            val_history['loss_regularization'].append(val_metrics['loss_regularization'])
            hybrid_model.train()

        current_lr = None
        if scheduler is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr
            current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % log_freq == 0:
            log_epoch_summary(epoch, num_epochs, epoch_metrics, val_metrics, lambda_current)

        log_tensorboard_epoch(writer, epoch, epoch_metrics, val_metrics, lambda_current,
                              grad_norm=epoch_metrics['grad_norm'], current_lr=current_lr)

        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                checkpoint_base_dir / f"epoch_{epoch+1}.pt",
                epoch, hybrid_model, optimizer, scheduler,
                lambda_current, train_history, val_history,
                norm_scale=denorm_factors if norm_type is not None else None,
                norm_offset=min_bounds    if norm_type is not None else None,
                arch_config=net_config,
                norm_type=norm_type,
            )

    print(f"Training complete | Final λ: {lambda_current:.4f}")

    final_path = checkpoint_base_dir / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        final_path, num_epochs - 1, hybrid_model, optimizer, scheduler,
        lambda_current, train_history, val_history,
        norm_scale=denorm_factors if norm_type is not None else None,
        norm_offset=min_bounds    if norm_type is not None else None,
        arch_config=net_config,
        norm_type=norm_type,
    )

    writer.close()
    print(f"TensorBoard logs: tensorboard --logdir {log_base}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid physics-augmented world model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)
