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
from pathlib import Path
from typing import Dict, Tuple
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


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trajectory_data(csv_path: str, config: Dict) -> Tuple:
    """
    Load trajectory data from CSV and prepare training/validation/test sets.
    
    Data loading strategy:
    - Read CSV and group by trajectory_id
    - Extract H-step sequences from each trajectory
    - Bias sampling toward early steps (step < 500) where most action happens
    - Normalize states if configured
    - Split into train/val/test sets
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader) ready for training
    """
    
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
    
    # Define column names (CSV has s_t_0 through s_t_13, but we use first 8)
    state_cols = [f's_t_{i}' for i in range(state_dim)]
    action_cols = [f'a_t_{i}' for i in range(action_dim)]
    next_state_cols = [f's_t+1_{i}' for i in range(state_dim)]
    
    # Compute state normalization statistics (on full dataset)
    if normalize:
        state_mean = df[state_cols].mean().values
        state_std = df[state_cols].std().values + 1e-8
        print(f"State normalization: mean={state_mean[:3]}..., std={state_std[:3]}...")
    else:
        state_mean = np.zeros(state_dim)
        state_std = np.ones(state_dim)
    
    # Extract sequences from each trajectory
    trajectory_sequences = []
    
    for traj_id, group in df.groupby('trajectory_id'):
        group = group.sort_values('step_id').reset_index(drop=True)
        
        # Extract numpy arrays for this trajectory
        states = group[state_cols].values
        actions = group[action_cols].values
        next_states = group[next_state_cols].values
        step_ids = group['step_id'].values
        
        num_steps = len(states)
        
        # Extract sliding windows of length H
        for start_idx in range(num_steps - horizon):
            start_step = step_ids[start_idx]
            
            # Bias toward early steps: 70% of samples from step < 500
            if start_step < 500:
                include_prob = 0.7
            else:
                include_prob = 0.3
            
            if np.random.rand() < include_prob:
                seq_states = states[start_idx:start_idx + horizon]
                seq_actions = actions[start_idx:start_idx + horizon]
                seq_next_states = next_states[start_idx:start_idx + horizon]
                
                # Normalize if configured
                if normalize:
                    seq_states = (seq_states - state_mean) / state_std
                    seq_next_states = (seq_next_states - state_mean) / state_std
                
                trajectory_sequences.append({
                    'initial_state': seq_states[0].copy(),
                    'actions': seq_actions.copy(),
                    'states': seq_next_states.copy(),
                })
    
    print(f"Extracted {len(trajectory_sequences)} trajectory sequences of length {horizon}")
    
    # Simple split: 70% train, 15% val, 15% test
    n_train = int(len(trajectory_sequences) * config['data']['train_fraction'])
    n_val = int(len(trajectory_sequences) * config['data']['val_fraction'])
    
    train_seqs = trajectory_sequences[:n_train]
    val_seqs = trajectory_sequences[n_train:n_train + n_val]
    test_seqs = trajectory_sequences[n_train + n_val:]
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test")
    
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
    
    return train_loader, val_loader, test_loader


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
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True
    )
    hybrid_model = hybrid_model.to(device)
    print(f"  ✓ Initialized (ds/dt = F_p + F_a)")
    print(f"  ✓ Model moved to {device}")
    
    return residual_network, hybrid_model


def create_optimizer(residual_network: PhysicsAugmented, config: Dict) -> torch.optim.Optimizer:
    """
    Create optimizer for residual network parameters.
    
    We only optimize the residual network weights. The physics prior
    is frozen and provides fixed baseline predictions.
    """
    train_config = config['training']
    
    # Adam optimizer with configured learning rate
    optimizer = torch.optim.Adam(
        residual_network.parameters(),
        lr=train_config['learning_rate']
    )
    
    print(f"Created Adam optimizer with lr={train_config['learning_rate']}")
    return optimizer


def main():
    """Main training script entrypoint."""
    print("\n" + "="*80)
    print("HYBRID PHYSICS-AUGMENTED WORLD MODEL TRAINING")
    print("="*80)
    
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
    optimizer = create_optimizer(residual_network, config)
    
    # Load trajectory data from CSV
    csv_path = Path(__file__).parent.parent / "data" / "updated_trajectory_data_noatmo.csv"
    train_loader, val_loader, test_loader = load_trajectory_data(str(csv_path), config)
    
    # Extract training hyperparameters
    train_config = config['training']
    aphynity_config = config['aphynity']
    
    num_epochs = train_config['num_epochs']
    horizon = train_config['horizon']
    batch_size = train_config['batch_size']
    log_freq = config['logging'].get('log_freq', 10)
    val_freq = train_config.get('val_freq', 5)
    checkpoint_freq = train_config.get('checkpoint_freq', 20)
    
    lambda_current = aphynity_config['lambda_init']
    tau_2 = aphynity_config['tau_2']
    lambda_min = aphynity_config['lambda_min']
    lambda_max = aphynity_config['lambda_max']
    
    # Initialize TensorBoard writer with timestamped directory
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
    print(f"  • Learning rate: {train_config['learning_rate']}")
    print(f"  • Grad clipping: {train_config['grad_clip_norm']}")
    
    print("\nAPHYNITY Objective:")
    print(f"  • Prediction horizon: {horizon} steps (~{horizon*config['integration']['dt']:.2f}s)")
    print(f"  • Initial λ: {lambda_current}")
    print(f"  • λ step size (τ_2): {tau_2}")
    print(f"  • λ bounds: [{lambda_min}, {lambda_max}]")
    
    # Training tracking
    from training_objective import train_aphynity_epoch
    
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
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # ========================================================================
    # TRAINING LOOP - Following the APHYNITY pseudocode
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    # Global step counter for per-batch TensorBoard logging
    global_step = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        # ====================================================================
        # Training epoch: process all batches
        # ====================================================================
        hybrid_model.train()
        epoch_metrics = {
            'loss_total': 0,
            'loss_trajectory': 0,
            'loss_regularization': 0,
            'batch_count': 0,
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
                ode_atol=config['integration']['atol']
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
            global_step += 1
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'L_total': f"{metrics['loss_total']:.4f}",
                'L_traj': f"{metrics['loss_trajectory']:.4f}",
                'λ': f"{lambda_current:.4f}"
            })
        
        # Average metrics over batches
        for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
            epoch_metrics[key] /= epoch_metrics['batch_count']
        
        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)
        
        # Log epoch-level metrics to TensorBoard (aggregated statistics)
        writer.add_scalar('Epoch/train_loss_total', epoch_metrics['loss_total'], epoch)
        writer.add_scalar('Epoch/train_loss_trajectory', epoch_metrics['loss_trajectory'], epoch)
        writer.add_scalar('Epoch/train_loss_regularization', epoch_metrics['loss_regularization'], epoch)
        writer.add_scalar('Epoch/lambda_final', lambda_current, epoch)
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
                    ode_module = HybridDynamicsODE(hybrid_model, device).to(device)
                    
                    for step in range(horizon):
                        action = batch_actions[:, step, :]
                        residual_output = hybrid_model.residual_network(current_state, action)
                        residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
                        residual_norms.append(residual_norm)
                        
                        # Set action and integrate using stable wrapper
                        ode_module.set_action(action)
                        t_eval = torch.tensor([0.0, 0.01], dtype=current_state.dtype, device=device)
                        solution = odeint(ode_module, current_state, t_eval, 
                                        method=config['integration']['method'],
                                        rtol=config['integration']['rtol'],
                                        atol=config['integration']['atol'])
                        next_state = solution[-1]
                        next_state = next_state.clamp(-100.0, 100.0)  # Clamp like training
                        predicted_states.append(next_state)
                        current_state = next_state
                    
                    predicted_trajectory = torch.stack(predicted_states, dim=1)
                    prediction_error = predicted_trajectory - batch_states
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
            
            # Early stopping check
            if val_metrics['loss_total'] < best_val_loss:
                best_val_loss = val_metrics['loss_total']
                patience_counter = 0
            else:
                patience_counter += 1
        
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
            checkpoint_path = Path(config['logging']['checkpoint_dir']) / f"epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'residual_state': hybrid_model.residual_network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lambda': lambda_current,
                'train_history': train_history,
                'val_history': val_history,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= config['regularization'].get('early_stopping_patience', 20):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final λ: {lambda_current:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_path = Path(config['logging']['checkpoint_dir']) / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hybrid_model.residual_network.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"  → Run: tensorboard --logdir {log_base}")
    print("  → Then open http://localhost:6006 in your browser")


if __name__ == '__main__':
    main()
