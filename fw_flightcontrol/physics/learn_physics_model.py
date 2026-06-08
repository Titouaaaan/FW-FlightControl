#!/usr/bin/env python3
import torch
torch.set_float32_matmul_precision('high')
import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import train_aphynity_batch, HybridDynamicsODE, rollout
from fw_flightcontrol.physics.utils import (
    load_config, load_trajectory_data, get_norm_type,
    log_epoch_summary, log_tensorboard_epoch, save_checkpoint,
    clean_state_dict_for_compilation,
)


def initialize_models(config: Dict, device: torch.device):
    net_config = config['network']
    prev_action_dim = net_config.get('prev_action_dim', 0)

    physics_prior = PhysicsPrior()
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        prev_action_dim=prev_action_dim,
        hidden_dims=net_config['hidden_dims'],
        activation=net_config['activation'],
    )
    num_params = sum(p.numel() for p in residual_network.parameters())
    input_size = net_config['state_dim'] + net_config['action_dim'] + prev_action_dim
    arch = f"{input_size} -> " + " -> ".join(map(str, net_config['hidden_dims'])) + f" -> {net_config['state_dim']}"
    print(f"Residual network: {arch}  ({num_params:,} params)")

    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
    ).to(device)
    return residual_network, hybrid_model


def load_checkpoint(checkpoint_path: str, residual_network, optimizer, scheduler, device) -> Dict:
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    residual_network.load_state_dict(
        clean_state_dict_for_compilation(checkpoint['residual_state'])
    )
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    print(f"  Resuming from epoch {checkpoint['epoch'] + 1}, λ={checkpoint['lambda']:.6f}")
    return {
        'start_epoch':    checkpoint['epoch'] + 1,
        'lambda_current': checkpoint['lambda'],
        'train_history':  checkpoint['train_history'],
        'val_history':    checkpoint['val_history'],
    }


def create_optimizer(residual_network, config: Dict):
    aphynity_config = config['aphynity']
    train_config    = config['training']
    lr = aphynity_config['tau_1']
    optimizer = torch.optim.Adam(residual_network.parameters(), lr=lr, betas=(0.9, 0.999))
    print(f"Adam optimizer: lr={lr}")

    scheduler, min_lr = None, None
    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('enabled', False):
        min_lr = scheduler_config.get('min_lr', 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config['num_epochs'], eta_min=min_lr,
        )
        print(f"CosineAnnealingLR: T_max={train_config['num_epochs']}, eta_min={min_lr}")

    return optimizer, scheduler, min_lr


def run_validation_epoch(
    ode_module: HybridDynamicsODE,
    t_eval: torch.Tensor,
    val_loader,
    config: Dict,
    device: torch.device,
    lambda_current: float,
    epoch: int,
    num_epochs: int,
) -> Dict:
    horizon = config['training']['horizon']
    ode_module.model.eval()
    totals = {'loss_total': 0.0, 'loss_trajectory': 0.0, 'loss_regularization': 0.0, 'n': 0}

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Val",
                               leave=False, unit="batch"):
            initial_states = batch_data['initial_states'].to(device)
            actions        = batch_data['actions'].to(device)
            ground_truth   = batch_data['states'].to(device)
            prev_actions   = batch_data.get('prev_actions')
            if prev_actions is not None:
                prev_actions = prev_actions.to(device)

            predicted_trajectory = rollout(
                ode_module, t_eval, initial_states, actions, prev_actions, horizon,
                config['integration']['method'], config['integration']['rtol'], config['integration']['atol'],
            )

            traj_loss = ((predicted_trajectory - ground_truth) ** 2).mean().item()

            B, H, S = ground_truth.shape
            residual_gt = ode_module.model.residual_network(
                ground_truth.view(B * H, S),
                actions.view(B * H, -1),
                prev_actions.view(B * H, -1) if prev_actions is not None else None,
            )
            reg_loss = ((residual_gt * ode_module.denorm_factors) ** 2).sum(dim=1).mean().item()

            totals['loss_trajectory']     += traj_loss
            totals['loss_regularization'] += reg_loss
            totals['loss_total']          += reg_loss + lambda_current * traj_loss
            totals['n']                   += 1

    n = totals.pop('n')
    return {k: v / n for k, v in totals.items()}


def main(resume_checkpoint: Optional[str] = None):
    config_path = Path(__file__).parent / 'training_params.yaml'
    config = load_config(str(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    residual_network, hybrid_model = initialize_models(config, device)
    hybrid_model.physics_prior    = torch.compile(hybrid_model.physics_prior)
    hybrid_model.residual_network = torch.compile(hybrid_model.residual_network)

    optimizer, scheduler, min_lr = create_optimizer(residual_network, config)

    start_epoch, resume_state = 0, None
    if resume_checkpoint:
        resume_state = load_checkpoint(resume_checkpoint, residual_network, optimizer, scheduler, device)
        start_epoch  = resume_state['start_epoch']

    csv_path = Path(__file__).parent.parent / "data" / config['data']['file']
    train_loader, val_loader, _, denorm_factors, min_bounds = load_trajectory_data(str(csv_path), config)
    denorm_factors_torch = torch.tensor(denorm_factors, dtype=torch.float32, device=device)
    min_bounds_torch     = torch.tensor(min_bounds,     dtype=torch.float32, device=device)
    norm_type = get_norm_type(config)

    if norm_type == 'data_driven_normalization':
        print(f"Normalization — std:  {denorm_factors.tolist()}")
        print(f"             — mean: {min_bounds.tolist()}")

    train_config    = config['training']
    aphynity_config = config['aphynity']
    num_epochs      = train_config['num_epochs']
    horizon         = train_config['horizon']
    log_freq        = config['logging'].get('log_freq', 10)
    val_freq        = train_config.get('val_freq', 5)
    checkpoint_freq = train_config.get('checkpoint_freq', 20)
    tau_2           = aphynity_config['tau_2']
    lambda_min      = aphynity_config['lambda_min']
    lambda_max      = aphynity_config['lambda_max']
    lambda_current  = resume_state['lambda_current'] if resume_state else aphynity_config['lambda_init']

    checkpoint_subdir   = config['logging'].get('checkpoint_subdirectory', 'checkpoints')
    run_name            = f"{datetime.now().strftime('%y%m%d')}_{checkpoint_subdir}"
    log_dir             = Path(__file__).parent.parent / "logs" / "tensorboard" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer              = SummaryWriter(str(log_dir))
    checkpoint_base_dir = Path(__file__).parent / config['logging']['checkpoint_dir'] / run_name

    if resume_state:
        train_history = resume_state['train_history']
        val_history   = resume_state['val_history']
    else:
        train_history = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': [], 'lambda_history': []}
        val_history   = {'loss_total': [], 'loss_trajectory': [], 'loss_regularization': []}

    print(f"\nTraining epochs {start_epoch}–{num_epochs-1} | horizon={horizon} | "
          f"lr={aphynity_config['tau_1']} | τ₂={tau_2} | λ₀={lambda_current}")

    global_step = 0
    config_yaml = yaml.dump(config, default_flow_style=False)

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch"):
        if epoch == start_epoch:
            writer.add_text('Config/full_config', config_yaml)

        # Build ODE module once per epoch — reused across all batches
        ode_module = HybridDynamicsODE(
            hybrid_model, device,
            denorm_factors=denorm_factors_torch,
            min_bounds=min_bounds_torch,
            norm_type=norm_type,
        ).to(device)
        t_eval = torch.tensor([0.0, config['integration']['dt']], dtype=torch.float32, device=device)

        hybrid_model.train()
        epoch_metrics = {'loss_total': 0.0, 'loss_trajectory': 0.0,
                         'loss_regularization': 0.0, 'grad_norm': 0.0, 'batch_count': 0}
        nan_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}: Train",
                    leave=False, unit="batch")
        for batch_data in pbar:
            try:
                metrics = train_aphynity_batch(
                    ode_module=ode_module,
                    t_eval=t_eval,
                    trajectory_batch=batch_data,
                    optimizer=optimizer,
                    lambda_current=lambda_current,
                    grad_clip_norm=train_config['grad_clip_norm'],
                    device=device,
                    ode_method=config['integration']['method'],
                    ode_rtol=config['integration']['rtol'],
                    ode_atol=config['integration']['atol'],
                )
            except ValueError as e:
                nan_batches += 1
                optimizer.zero_grad()
                tqdm.write(f"  [Epoch {epoch+1}] Skipped NaN batch ({nan_batches} so far): {e}")
                continue

            for k in ['loss_total', 'loss_trajectory', 'loss_regularization', 'grad_norm']:
                epoch_metrics[k] += metrics[k]
            epoch_metrics['batch_count'] += 1

            # Per-batch lambda update per APHYNITY paper: λ_{j+1} = λ_j + τ₂ * L_traj
            lambda_current = max(lambda_min, min(
                lambda_current + tau_2 * metrics['loss_trajectory'], lambda_max
            ))

            writer.add_scalar('Batch/loss_total',          metrics['loss_total'],          global_step)
            writer.add_scalar('Batch/loss_trajectory',     metrics['loss_trajectory'],     global_step)
            writer.add_scalar('Batch/loss_regularization', metrics['loss_regularization'], global_step)
            writer.add_scalar('Batch/lambda',              lambda_current,                 global_step)
            global_step += 1
            pbar.set_postfix({'L_traj': f"{metrics['loss_trajectory']:.4f}", 'λ': f"{lambda_current:.4f}"})

        if nan_batches:
            tqdm.write(f"  [Epoch {epoch+1}] {nan_batches} NaN batches skipped")
        if epoch_metrics['batch_count'] == 0:
            tqdm.write(f"  [Epoch {epoch+1}] WARNING: all batches were NaN — skipping epoch")
            continue

        n = epoch_metrics.pop('batch_count')
        for k in epoch_metrics:
            epoch_metrics[k] /= n

        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)

        val_metrics = None
        if (epoch + 1) % val_freq == 0:
            val_metrics = run_validation_epoch(
                ode_module, t_eval, val_loader, config, device, lambda_current, epoch, num_epochs,
            )
            val_history['loss_total'].append(val_metrics['loss_total'])
            val_history['loss_trajectory'].append(val_metrics['loss_trajectory'])
            val_history['loss_regularization'].append(val_metrics['loss_regularization'])
            hybrid_model.train()

        if scheduler is not None:
            scheduler.step()
            if min_lr is not None:
                for pg in optimizer.param_groups:
                    if pg['lr'] < min_lr:
                        pg['lr'] = min_lr

        current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else None

        if (epoch + 1) % log_freq == 0:
            log_epoch_summary(epoch, num_epochs, epoch_metrics, val_metrics, lambda_current)
        log_tensorboard_epoch(writer, epoch, epoch_metrics, val_metrics, lambda_current,
                              epoch_metrics['grad_norm'], current_lr)

        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                checkpoint_base_dir / f"epoch_{epoch+1}.pt",
                epoch, hybrid_model, optimizer, scheduler,
                lambda_current, train_history, val_history,
                norm_scale=denorm_factors, norm_offset=min_bounds,
            )

    save_checkpoint(
        checkpoint_base_dir / "final_model.pt",
        num_epochs - 1, hybrid_model, optimizer, scheduler,
        lambda_current, train_history, val_history,
        norm_scale=denorm_factors, norm_offset=min_bounds,
    )
    writer.close()
    print(f"Done. λ_final={lambda_current:.4f} | TensorBoard: tensorboard --logdir {log_dir.parent}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)
