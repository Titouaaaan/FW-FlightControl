#!/usr/bin/env python3
"""
Training script for pendulum residual network using APHYNITY.

This script implements residual learning on the pendulum with friction:
1. Generate diverse trajectories from the gym pendulum environment (with friction)
2. Train a small MLP residual network to learn friction dynamics
3. The physics prior (frictionless pendulum) provides the base dynamics
4. APHYNITY objective learns the residual: F_residual = F_true - F_prior

Key insights:
- Physics prior alone fails with friction (RMSE grows 0.007→6.0 over 40 steps)
- Residual network learns to correct for the -0.2*omega damping term
- Semi-implicit Euler is used for stable, energy-conserving integration
- Data is generated on-the-fly from environment (no disk storage needed)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics.pendulum.pendulum_physics import PendulumPhysics
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import train_aphynity_epoch
from torchdiffeq import odeint


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_trajectories(
    num_trajectories=100,
    max_horizon=40,
    friction_alpha=0.2,
    device='cpu'
):
    """
    Generate diverse pendulum trajectories with friction.
    
    Args:
        num_trajectories: Number of trajectories to generate
        max_horizon: Maximum steps per trajectory
        friction_alpha: Friction coefficient (0.2 for significant damping)
        device: torch device
    
    Returns:
        trajectories: list of dicts with 'states', 'actions', 'next_states'
    """
    env = gym.make("Pendulum-v1")
    physics_prior = PendulumPhysics(omega0_square=15.0, alpha=friction_alpha).to(device)
    
    trajectories = []
    dt = 0.05
    
    print(f"\nGenerating {num_trajectories} trajectories with friction (alpha={friction_alpha})...")
    
    for traj_id in tqdm(range(num_trajectories), desc="Trajectory generation", unit="traj"):
        # Random initialization
        obs, _ = env.reset()
        theta = np.arctan2(obs[1], obs[0])
        omega = obs[2]
        state = torch.tensor([[theta, omega]], dtype=torch.float32, device=device)
        
        # Random trajectory length
        horizon = np.random.randint(10, max_horizon + 1)
        
        states = [state.clone()]
        actions = []
        next_states = []
        
        for step in range(horizon):
            # Generate diverse actions (mostly zero, some random exploration)
            if np.random.rand() < 0.7:
                action = torch.zeros(1, 1, device=device)
            else:
                action = torch.tensor([[np.random.uniform(-1, 1)]], dtype=torch.float32, device=device)
            
            # Single step integration using semi-implicit Euler
            theta = state[:, 0:1]
            omega = state[:, 1:2]
            
            with torch.no_grad():
                derivatives = physics_prior(state, action)
                dtheta_dt = derivatives[:, 0:1]
                domega_dt = derivatives[:, 1:2]
                
                omega_new = omega + domega_dt * dt
                theta_new = theta + omega_new * dt
                
                next_state = torch.cat([theta_new, omega_new], dim=1)
                next_state = next_state.clamp(-3 * np.pi, 3 * np.pi)  # Clamp angles
            
            states.append(next_state.clone())
            actions.append(action.clone())
            next_states.append(next_state.clone())
            state = next_state
        
        # Convert to cpu numpy for storage
        trajectories.append({
            'states': torch.stack(states[:-1]).squeeze().cpu().numpy(),  # (H, 2)
            'actions': torch.stack(actions).squeeze().cpu().numpy(),     # (H, 1) or (H,)
            'next_states': torch.stack(next_states).squeeze().cpu().numpy(),  # (H, 2)
        })
    
    env.close()
    print(f"Generated {len(trajectories)} trajectories")
    
    return trajectories


# ============================================================================
# DATA LOADING
# ============================================================================

class PendulumTrajectoryDataset(Dataset):
    """
    Dataset that extracts fixed-horizon subsequences from trajectories.
    
    During training, we sample H-step subsequences to form mini-batches.
    This is more computationally efficient than unrolling entire trajectories.
    """
    
    def __init__(self, trajectories, horizon=30, normalize=False):
        self.trajectories = trajectories
        self.horizon = horizon
        self.normalize = normalize
        
        # Collect normalization statistics
        if normalize:
            all_states = np.concatenate([t['states'] for t in trajectories], axis=0)
            self.state_mean = all_states.mean(axis=0)
            self.state_std = all_states.std(axis=0) + 1e-8
        else:
            self.state_mean = np.zeros(2)
            self.state_std = np.ones(2)
        
        # Extract all valid H-length sequences
        self.sequences = []
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            next_states = traj['next_states']
            
            # Ensure action is 2D
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            if next_states.ndim == 1:
                next_states = next_states.reshape(-1, 1)
            
            # Extract H-length sliding windows
            T = len(states)
            if T >= self.horizon:  # Only extract if trajectory is long enough
                for start_idx in range(T - self.horizon + 1):
                    end_idx = start_idx + self.horizon
                    seq_initial_state = states[start_idx]
                    seq_actions = actions[start_idx:end_idx]
                    seq_next_states = next_states[start_idx:end_idx]
                    
                    # Normalize if needed
                    if normalize:
                        seq_initial_state = (seq_initial_state - self.state_mean) / self.state_std
                        seq_next_states = (seq_next_states - self.state_mean) / self.state_std
                    
                    self.sequences.append({
                        'initial_state': torch.tensor(seq_initial_state, dtype=torch.float32),
                        'actions': torch.tensor(seq_actions, dtype=torch.float32),
                        'next_states': torch.tensor(seq_next_states, dtype=torch.float32),
                    })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'initial_states': seq['initial_state'],
            'actions': seq['actions'],
            'states': seq['next_states'],
        }


def create_data_loaders(trajectories, horizon=30, batch_size=32, train_fraction=0.8):
    """
    Create train/val data loaders from trajectories.
    
    Returns:
        (train_loader, val_loader)
    """
    dataset = PendulumTrajectoryDataset(trajectories, horizon=horizon, normalize=False)
    
    # Split into train/val
    train_size = int(len(dataset) * train_fraction)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    def collate_fn(batch):
        """Collate batch of sequences."""
        initial_states = torch.stack([b['initial_states'] for b in batch])
        actions = torch.stack([b['actions'] for b in batch])
        states = torch.stack([b['states'] for b in batch])
        return {
            'initial_states': initial_states,
            'actions': actions,
            'states': states,
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"\nData split: {train_size} train sequences, {val_size} val sequences")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# ============================================================================
# HYBRID DYNAMICS MODEL (Prior + Residual)
# ============================================================================

class PendulumHybridDynamics(nn.Module):
    """
    Hybrid model combining frictionless physics prior + residual network.
    
    ds/dt = F_prior(s) + F_residual(s, a)
    """
    
    def __init__(self, physics_prior, residual_network, device='cpu'):
        super().__init__()
        self.physics_prior = physics_prior
        self.residual_network = residual_network
        self.device = device
    
    def forward(self, state, action):
        """
        Compute hybrid dynamics.
        
        Args:
            state: (batch_size, 2)
            action: (batch_size, 1)
        
        Returns:
            derivatives: (batch_size, 2)
        """
        prior_deriv = self.physics_prior(state, action)
        residual_deriv = self.residual_network(state, action)
        return prior_deriv + residual_deriv


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    hybrid_model,
    residual_network,
    train_loader,
    optimizer,
    device,
    lambda_current,
    tau_1,
    tau_2,
    writer,
    global_step,
    horizon=30
):
    """
    Train for one epoch using APHYNITY objective.
    
    Returns: (epoch_metrics, lambda_current, global_step)
    """
    hybrid_model.train()
    epoch_metrics = {
        'loss_total': 0.0,
        'loss_trajectory': 0.0,
        'loss_regularization': 0.0,
        'batch_count': 0,
    }
    lambda_history = []
    
    pbar = tqdm(train_loader, desc="Training batch", leave=False)
    
    for batch in pbar:
        # Use train_aphynity_epoch to compute loss and gradients
        # Using semi-implicit Euler for stable, symplectic integration in both train and val
        metrics = train_aphynity_epoch(
            hybrid_model=hybrid_model,
            trajectory_batch=batch,
            optimizer=optimizer,
            lambda_current=lambda_current,
            tau_1=tau_1,
            tau_2=tau_2,
            device=device,
            ode_method='semi_implicit_euler',  # Symplectic integrator for pendulum
            ode_rtol=1e-4,
            ode_atol=1e-5
        )
        
        # Accumulate epoch metrics
        epoch_metrics['loss_total'] += metrics['loss_total']
        epoch_metrics['loss_trajectory'] += metrics['loss_trajectory']
        epoch_metrics['loss_regularization'] += metrics['loss_regularization']
        epoch_metrics['batch_count'] += 1
        
        lambda_current = metrics['lambda_new']
        lambda_history.append(lambda_current)
        
        # Log batch-level metrics to TensorBoard
        writer.add_scalar('Batch/loss_total', metrics['loss_total'], global_step)
        writer.add_scalar('Batch/loss_trajectory', metrics['loss_trajectory'], global_step)
        writer.add_scalar('Batch/loss_regularization', metrics['loss_regularization'], global_step)
        writer.add_scalar('Batch/lambda', lambda_current, global_step)
        
        global_step += 1
        pbar.set_postfix({
            'L_total': f"{metrics['loss_total']:.4f}",
            'L_traj': f"{metrics['loss_trajectory']:.4f}",
            'λ': f"{lambda_current:.4f}"
        })
    
    # Average metrics over batches
    for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
        epoch_metrics[key] /= epoch_metrics['batch_count']
    
    return epoch_metrics, lambda_current, global_step


@torch.no_grad()
def validate(
    hybrid_model,
    val_loader,
    device,
    writer,
    epoch,
    horizon=30
):
    """Validate on held-out trajectories."""
    hybrid_model.eval()
    val_metrics = {
        'loss_total': 0.0,
        'loss_trajectory': 0.0,
        'loss_regularization': 0.0,
        'batch_count': 0,
    }
    
    pbar = tqdm(val_loader, desc="Validation batch", leave=False)
    
    for batch in pbar:
        initial_states = batch['initial_states'].to(device)  # (B, 2)
        actions = batch['actions'].to(device)  # (B, H, 1)
        ground_truth_states = batch['states'].to(device)  # (B, H, 2)
        
        # Unroll trajectory using hybrid model
        predicted_states = []
        residual_norms = []
        current_state = initial_states
        
        for step in range(horizon):
            action = actions[:, step, :]
            
            # Compute residual norm
            residual_output = hybrid_model.residual_network(current_state, action)
            residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
            residual_norms.append(residual_norm)
            
            # Step forward with semi-implicit Euler
            next_state = hybrid_model.integrate(current_state, action, dt=0.05)
            predicted_states.append(next_state)
            current_state = next_state
        
        predicted_trajectory = torch.stack(predicted_states, dim=1)  # (B, H, 2)
        
        # Compute losses
        trajectory_loss = torch.norm(predicted_trajectory - ground_truth_states, p=2, dim=2).mean()
        regularization_loss = torch.stack(residual_norms).mean()
        
        # Metrics for logging
        batch_loss_total = regularization_loss.item() + trajectory_loss.item()
        val_metrics['loss_total'] += batch_loss_total
        val_metrics['loss_trajectory'] += trajectory_loss.item()
        val_metrics['loss_regularization'] += regularization_loss.item()
        val_metrics['batch_count'] += 1
        
        pbar.set_postfix({'L_total': f"{batch_loss_total:.4f}"})
    
    # Average over batches
    for key in ['loss_total', 'loss_trajectory', 'loss_regularization']:
        val_metrics[key] /= val_metrics['batch_count']
    
    # Log validation metrics to TensorBoard
    writer.add_scalar('Epoch/val_loss_total', val_metrics['loss_total'], epoch)
    writer.add_scalar('Epoch/val_loss_trajectory', val_metrics['loss_trajectory'], epoch)
    writer.add_scalar('Epoch/val_loss_regularization', val_metrics['loss_regularization'], epoch)
    
    return val_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training script."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Hyperparameters
    num_trajectories = 800  # Increased significantly for better generalization and data diversity
    horizon = 15  # Shorter horizon reduces multi-step error compounding
    batch_size = 32
    num_epochs = 75
    learning_rate = 1e-3
    friction_alpha = 0.2  # Friction that residual must learn
    
    print("\n" + "="*80)
    print("PENDULUM RESIDUAL LEARNING - APHYNITY STYLE")
    print("="*80)
    
    # Generate trajectories
    trajectories = generate_trajectories(
        num_trajectories=num_trajectories,
        max_horizon=horizon + 20,  # Generate longer trajectories than needed for better sequence extraction
        friction_alpha=friction_alpha,
        device=device
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        trajectories,
        horizon=horizon,
        batch_size=batch_size,
        train_fraction=0.8
    )
    
    # Initialize models
    print("\n" + "="*80)
    print("INITIALIZING MODELS")
    print("="*80)
    
    physics_prior = PendulumPhysics(omega0_square=15.0, alpha=0.0).to(device)
    print("\nPhysics Prior: Frictionless (alpha=0.0)")
    print("  Status: FROZEN (non-trainable)")
    
    residual_network = PhysicsAugmented(
        state_dim=2,
        action_dim=1,
        hidden_dims=[64, 64],
        activation='relu',
        use_batch_norm=False
    ).to(device)
    num_params = sum(p.numel() for p in residual_network.parameters())
    print(f"\nResidual Network (PhysicsAugmented): [3] -> [64, 64] -> [2]")
    print(f"  Parameters: {num_params:,} (trainable)")
    
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True,
        integration_method='semi_implicit_euler'  # Use symplectic integrator for pendulum
    ).to(device)
    print(f"\nHybrid Model: ds/dt = F_prior + F_residual")
    print(f"  Integration: Semi-implicit Euler (dt=0.05s)")
    
    # APHYNITY hyperparameters (from official repo for pendulum)
    tau_1 = 1e-3  # Learning rate for optimizer (NOT gradient scaling!)
    tau_2 = 1.0  # Large lambda step size (official APHYNITY uses 1.0 for pendulum)
    lambda_init = 1.0  # Initial Lagrange multiplier
    
    # Optimizer (only for residual network; physics prior is frozen)
    # Use tau_1 as learning rate (official APHYNITY approach)
    optimizer = optim.Adam(residual_network.parameters(), lr=tau_1, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # TensorBoard logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent / "logs" / "tensorboard" / f"pendulum_residual_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"\nTensorBoard logs: {log_dir}")
    
    # Log config
    config_str = f"""
# Pendulum Residual Learning Configuration

## Environment
- Friction coefficient (alpha): {friction_alpha}
- Environment: Gym Pendulum-v1 with friction
- Trajectories generated: {num_trajectories}

## Training
- Epochs: {num_epochs}
- Batch size: {batch_size}
- Learning rate: {learning_rate}
- Horizon: {horizon} steps (~{horizon*0.05:.2f}s)
- Integration: Semi-implicit Euler (dt=0.05s)

## Architecture
- State dim: 2 (theta, omega)
- Action dim: 1 (torque)
- Residual MLP: 3 -> [64, 64] -> 2
- Total params: {num_params:,}

## APHYNITY Parameters
- tau_1 (gradient scaling): {tau_1}
- tau_2 (lambda step size): {tau_2}
- lambda_init: {lambda_init}

## Objective
- Loss: L = ||F_a|| + λ * ||trajectory_error||
- Trains residual network only
- Physics prior (frictionless) frozen
- Multi-step prediction error (error compounding)
"""
    writer.add_text('Config', config_str)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING (APHYNITY)")
    print("="*80)
    
    best_val_loss = float('inf')
    lambda_current = lambda_init
    global_step = 0
    
    # History tracking
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
    
    for epoch in range(num_epochs):
        # Training
        epoch_metrics, lambda_current, global_step = train_epoch(
            hybrid_model, residual_network,
            train_loader, optimizer,
            device, lambda_current, tau_1, tau_2,
            writer, global_step,
            horizon=horizon
        )
        
        # Record training history
        train_history['loss_total'].append(epoch_metrics['loss_total'])
        train_history['loss_trajectory'].append(epoch_metrics['loss_trajectory'])
        train_history['loss_regularization'].append(epoch_metrics['loss_regularization'])
        train_history['lambda_history'].append(lambda_current)
        
        # Log epoch-level training metrics
        writer.add_scalar('Epoch/train_loss_total', epoch_metrics['loss_total'], epoch)
        writer.add_scalar('Epoch/train_loss_trajectory', epoch_metrics['loss_trajectory'], epoch)
        writer.add_scalar('Epoch/train_loss_regularization', epoch_metrics['loss_regularization'], epoch)
        writer.add_scalar('Epoch/lambda_final', lambda_current, epoch)
        
        # Validation
        val_metrics = validate(
            hybrid_model, val_loader,
            device, writer, epoch,
            horizon=horizon
        )
        
        # Record validation history
        val_history['loss_total'].append(val_metrics['loss_total'])
        val_history['loss_trajectory'].append(val_metrics['loss_trajectory'])
        val_history['loss_regularization'].append(val_metrics['loss_regularization'])
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/learning_rate', current_lr, epoch)
        
        # Save best checkpoint
        if val_metrics['loss_total'] < best_val_loss:
            best_val_loss = val_metrics['loss_total']
            checkpoint_path = log_dir / "best_model.pt"
            torch.save(residual_network.state_dict(), checkpoint_path)
            status = "✓"
        else:
            status = " "
        
        # Console logging
        log_str = f"[Epoch {epoch+1:3d}/{num_epochs}] {status} "
        log_str += f"Train: L_total={epoch_metrics['loss_total']:.4f} | "
        log_str += f"L_traj={epoch_metrics['loss_trajectory']:.4f} | "
        log_str += f"L_reg={epoch_metrics['loss_regularization']:.4f} | "
        log_str += f"λ={lambda_current:.4f} | "
        log_str += f"Val: L_total={val_metrics['loss_total']:.4f}"
        print(log_str)
    
    # Final checkpoint
    final_checkpoint_path = log_dir / "final_model.pt"
    torch.save(residual_network.state_dict(), final_checkpoint_path)
    print(f"\n✓ Training complete!")
    print(f"  Best model: {log_dir / 'best_model.pt'}")
    print(f"  Final model: {final_checkpoint_path}")
    print(f"  Logs: {log_dir}")
    
    writer.close()


if __name__ == "__main__":
    main()
