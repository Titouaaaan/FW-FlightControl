#!/usr/bin/env python3
import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import train_aphynity_epoch, HybridDynamicsODE
from fw_flightcontrol.physics.utils import (
    load_config, load_trajectory_data,
    normalize_state_torch, denormalize_state_torch,
    log_epoch_summary, log_tensorboard_epoch, save_checkpoint,
)

def initialize_models(config: Dict, device: torch.device) -> Tuple[PhysicsAugmented, HybridDynamicsModel]:
    physics_prior = PhysicsPrior()

    net_config = config['network']
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=net_config['hidden_dims'],
        activation=net_config['activation'],
        use_batch_norm=net_config['use_batch_norm'],
    )
    num_residual_params = sum(p.numel() for p in residual_network.parameters())
    input_size = net_config['state_dim'] + net_config['action_dim']
    arch_str = f"{input_size} -> " + " -> ".join(map(str, net_config['hidden_dims'])) + f" -> {net_config['state_dim']}"
    print(f"\nResidual Network:")
    print(f"  ✓ Created MLP with {num_residual_params:,} trainable parameters")
    print(f"  ✓ Architecture: {arch_str}")

    integration_method = config.get('integration', {}).get('method', 'rk4')
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True,
        integration_method=integration_method
    )
    hybrid_model = hybrid_model.to(device)

    return residual_network, hybrid_model


def load_checkpoint(checkpoint_path: str, residual_network: PhysicsAugmented, optimizer, scheduler, device: torch.device) -> Dict:
    """Load a checkpoint and restore training state."""
    from fw_flightcontrol.physics.utils import clean_state_dict_for_compilation
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    residual_state = clean_state_dict_for_compilation(checkpoint['residual_state'])
    residual_network.load_state_dict(residual_state)
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

    return {
        'start_epoch':    start_epoch,
        'lambda_current': lambda_current,
        'train_history':  train_history,
        'val_history':    val_history,
    }


def create_optimizer(residual_network: PhysicsAugmented, config: Dict):
    """Create Adam optimizer and optional StepLR scheduler for the residual network."""
    aphynity_config = config['aphynity']
    train_config    = config['training']

    learning_rate = aphynity_config['tau_1']
    optimizer = torch.optim.Adam(residual_network.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    scheduler = None
    min_lr    = None
    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('enabled', False):
        if scheduler_config.get('type') == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.7)
            )
            min_lr = scheduler_config.get('min_lr', 1e-5)

    return optimizer, scheduler, min_lr


def run_validation_epoch(
    hybrid_model: HybridDynamicsModel,
    val_loader,
    config: Dict,
    device: torch.device,
    lambda_current: float,
    epoch: int,
    num_epochs: int,
) -> Dict:
    """Run one full validation pass and return averaged loss metrics."""
    horizon    = config['training']['horizon']
    ode_method = config['integration']['method']
    dt         = config['integration']['dt']

    hybrid_model.eval()
    val_metrics = {
        'loss_total':          0,
        'loss_trajectory':     0,
        'loss_regularization': 0,
        'batch_count':         0,
    }

    ode_module = HybridDynamicsODE(hybrid_model, device).to(device)
    t_eval = torch.tensor([0.0, dt], dtype=torch.float32, device=device)

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
                current_state_raw = denormalize_state_torch(
                    current_state, hybrid_model.norm_scale, hybrid_model.norm_offset
                )
                ode_module.set_action(action)
                ode_module.arm_capture()

                if ode_method == 'semi_implicit_euler':
                    derivatives_raw = ode_module(None, current_state_raw)
                    state_dim = current_state_raw.shape[-1]
                    half_dim  = state_dim // 2
                    vel_new   = current_state_raw[:, half_dim:] + derivatives_raw[:, half_dim:] * dt
                    pos_new   = current_state_raw[:, :half_dim] + vel_new * dt
                    next_state_raw = torch.cat([pos_new, vel_new], dim=-1).clamp(-1000.0, 1000.0)
                else:
                    solution = odeint(ode_module, current_state_raw, t_eval,
                                      method=ode_method,
                                      rtol=config['integration']['rtol'],
                                      atol=config['integration']['atol'])
                    next_state_raw = solution[-1].clamp(-1000.0, 1000.0)

                if hybrid_model.with_residual and ode_module.captured_residual_norm is not None:
                    residual_norms.append(ode_module.captured_residual_norm)

                next_state = normalize_state_torch(
                    next_state_raw, hybrid_model.norm_scale, hybrid_model.norm_offset
                )
                predicted_states.append(next_state)
                current_state = next_state

            predicted_trajectory = torch.stack(predicted_states, dim=1)

            trajectory_loss     = torch.norm(predicted_trajectory - batch_states, p=2, dim=2).mean()
            regularization_loss = torch.stack(residual_norms).mean() if residual_norms \
                                  else torch.tensor(0.0, device=device)

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



def _print_training_config(config, num_epochs, batch_size, horizon, lambda_current, tau_2, lambda_min, lambda_max):
    aphynity_config = config['aphynity']
    train_config    = config['training']
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"\nTraining Loop:")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Learning rate: {aphynity_config['tau_1']}")
    print(f"  • Grad clipping: {train_config['grad_clip_norm']}")
    print(f"\nAPHYNITY Objective:")
    print(f"  • Prediction horizon: {horizon} steps (~{horizon*config['integration']['dt']:.2f}s)")
    print(f"  • Initial λ: {lambda_current}")
    print(f"  • λ step size (τ_2): {tau_2}")
    print(f"  • λ bounds: [{lambda_min}, {lambda_max}]")


def _build_hyperparams_text(config, batch_size, horizon, aphynity_config, train_config):
    return f"""
## Training Hyperparameters

**Learning:**
- Learning Rate: {aphynity_config['tau_1']}
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


def main(resume_checkpoint: Optional[str] = None):
    config_path = Path(__file__).parent / 'training_params.yaml'
    config = load_config(str(config_path))

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

    csv_path = Path(__file__).parent / "data" / "updated_trajectory_data_progressive_noatmo_2.0.csv"
    train_loader, val_loader, _, norm_scale, norm_offset = load_trajectory_data(str(csv_path), config)

    # Attach normalization parameters to the model so they travel with every checkpoint
    norm_scale_t  = torch.tensor(norm_scale,  dtype=torch.float32, device=device)
    norm_offset_t = torch.tensor(norm_offset, dtype=torch.float32, device=device)
    hybrid_model.norm_scale  = norm_scale_t
    hybrid_model.norm_offset = norm_offset_t

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

    _print_training_config(config, num_epochs, batch_size, horizon,
                           lambda_current, tau_2, lambda_min, lambda_max)

    if resume_state:
        train_history = resume_state['train_history']
        val_history   = resume_state['val_history']
        print(f"\nResumed training history: {len(train_history['loss_total'])} epochs completed")
    else:
        train_history = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': [], 'lambda_history': []}
        val_history   = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': []}

    global_step = 0
    config_yaml = yaml.dump(config, default_flow_style=False)

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch"):

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

        for batch_data in pbar:
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
            )

            epoch_metrics['loss_total']          += metrics['loss_total']
            epoch_metrics['loss_trajectory']     += metrics['loss_trajectory']
            epoch_metrics['loss_regularization'] += metrics['loss_regularization']
            epoch_metrics['batch_count']         += 1

            lambda_current = max(lambda_min, min(metrics['lambda_new'], lambda_max))

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

        for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
            epoch_metrics[key] /= epoch_metrics['batch_count']
        if epoch_metrics['batch_count'] > 0:
            epoch_metrics['grad_norm_before_clipping'] /= epoch_metrics['batch_count']
            epoch_metrics['grad_norm_after_clipping']  /= epoch_metrics['batch_count']

        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)

        val_metrics = None
        if (epoch + 1) % val_freq == 0:
            val_metrics = run_validation_epoch(
                hybrid_model, val_loader, config, device, lambda_current, epoch, num_epochs,
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

        grad_metrics = {k: epoch_metrics[k] for k in [
            'grad_norm_before_clipping', 'grad_norm_after_clipping',
        ]}
        log_tensorboard_epoch(writer, epoch, epoch_metrics, val_metrics, lambda_current, grad_metrics, current_lr)

        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                checkpoint_base_dir / f"epoch_{epoch+1}.pt",
                epoch, hybrid_model, optimizer, scheduler,
                lambda_current, train_history, val_history,
            )

    final_path = checkpoint_base_dir / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        final_path, num_epochs - 1, hybrid_model, optimizer, scheduler,
        lambda_current, train_history, val_history,
    )

    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid physics-augmented world model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)
