"""
Training objective functions for hybrid physics-residual world model.

This module defines three training strategies:
1. APHYNITY: Compounding multi-step prediction error over full trajectory
2. PhIHP: One-step prediction error only
3. Hybrid: H-step prediction with loss on final state only
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchdiffeq import odeint
from typing import Dict


def train_aphynity_epoch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    lambda_current: float,
    tau_2: float,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Train the residual network using APHYNITY (Augmented Physics model with Adaptive Residuals).
    
    APHYNITY trains on compounding multi-step prediction error over an H-step trajectory horizon.
    The residual network learns to correct the physics prior's errors, which cascade through the
    integration steps. This approach optimizes the combined model for long-horizon prediction.
    
    The objective is:
        L_λ(θ_a) = ||F_a^{θ_a}|| + λ · L_traj(θ_a)
    
    where:
        - ||F_a^{θ_a}|| is the L2 norm of residual network outputs (regularization)
        - L_traj = Σ_{k=1}^{H} ||ŝ_{t+k} - s_{t+k}||² (trajectory prediction error)
        - λ is the Lagrange multiplier (updated via dual ascent)
    
    Args:
        hybrid_model: HybridDynamicsModel containing both physics_prior and residual_network.
                     Only residual_network parameters will be optimized.
        trajectory_batch: Dictionary containing trajectory segments:
            - 'initial_states': Tensor of shape (batch_size, 8) - initial state s_t
            - 'actions': Tensor of shape (batch_size, H, 3) - actions a_t, ..., a_{t+H-1}
            - 'states': Tensor of shape (batch_size, H, 8) - ground truth s_{t+1}, ..., s_{t+H}
        optimizer: PyTorch optimizer configured for residual_network parameters only.
                  Physics prior parameters should not be included.
        lambda_current: Current Lagrange multiplier value λ_j for this epoch.
        tau_2: Step size for Lagrange multiplier update (dual ascent step size).
        device: Torch device (CPU or CUDA).
    
    Returns:
        Dictionary with epoch metrics:
            - 'loss_total': Combined loss value (λ term + regularization)
            - 'loss_trajectory': Trajectory prediction MSE across H steps
            - 'loss_regularization': Mean L2 norm of residual outputs
            - 'lambda_new': Updated Lagrange multiplier λ_{j+1} = λ_j + τ_2 · L_traj
    
    Algorithm:
        1. Forward pass: Predict H-step trajectory by repeated RK4 integration of F_p + F_a
        2. Compute regularization loss: mean of residual output norms at each step
        3. Compute trajectory loss: MSE between predictions and ground truth across all steps
        4. Backward pass: Backprop only through residual network (physics prior frozen)
        5. Update residual parameters via optimizer step
        6. Update Lagrange multiplier via dual ascent
    """
    
    # Move batch data to device
    initial_states = trajectory_batch['initial_states'].to(device)  # (batch_size, 8)
    actions = trajectory_batch['actions'].to(device)                # (batch_size, H, 3)
    ground_truth_states = trajectory_batch['states'].to(device)     # (batch_size, H, 8)
    
    batch_size = initial_states.shape[0]
    horizon = actions.shape[1]
    
    # Ensure model is on correct device
    hybrid_model = hybrid_model.to(device)
    
    # Zero gradients before forward pass
    optimizer.zero_grad()
    
    # === FORWARD PASS: H-step Trajectory Prediction ===
    predicted_states = []
    regularization_losses = []
    
    current_state = initial_states
    
    for step in range(horizon):
        action = actions[:, step, :]  # (batch_size, 3)
        
        # Compute residual output at step boundary for regularization term
        # F_a^{θ_a}(ŝ_{t+k-1}, a_{t+k-1}) - per the paper, only evaluated at step boundaries
        residual_at_boundary = hybrid_model.residual_network(current_state, action)
        
        # Regularization: L2 norm of residual output
        reg_loss_step = torch.norm(residual_at_boundary, p=2, dim=1).mean()
        regularization_losses.append(reg_loss_step)
        
        # === RK4 Integration: One simulation step (0.01 second) ===
        # Use the hybrid model's forward pass (which combines physics_prior + residual)
        # Integrate the combined dynamics: ŝ_{t+k} = ŝ_{t+k-1} + ∫_0^{Δt} (F_p + F_a)(s, a) dt
        def ode_dynamics(t, state_t):
            """Combined ODE: physics prior + learned residual."""
            return hybrid_model(state_t, action)
        
        # Time points for ODE integration
        t_eval = torch.tensor([0.0, 0.01], dtype=current_state.dtype, device=device)
        
        # Solve ODE over one step [0, 0.01] using RK4
        solution = odeint(ode_dynamics, current_state, t_eval, method='rk4')
        # solution shape: (len(t_eval), batch_size, state_dim) = (2, batch_size, 8)
        next_state = solution[-1]  # Select final time point for all batches
        
        predicted_states.append(next_state)
        current_state = next_state
    
    # Stack all predicted states: (batch_size, H, 8)
    predicted_trajectory = torch.stack(predicted_states, dim=1)
    
    # === LOSS COMPUTATION ===
    
    # Trajectory loss: L2 norm of prediction error at each step, averaged
    # L_traj = (1/NH) Σ_i^N Σ_k^H ||s_{t+k}^{(i)} - ŝ_{t+k}^{(i)}||
    error_vectors = predicted_trajectory - ground_truth_states  # (batch_size, H, 8)
    trajectory_loss = torch.norm(error_vectors, p=2, dim=2).mean()  # L2 norm per sample per step, then average
    
    # Regularization loss: mean L2 norm of residual outputs across trajectory
    # ||F_a|| = (1/H) Σ_k^H E_batch[||F_a(ŝ_{t+k-1}, a_{t+k-1})||]
    regularization_loss = torch.stack(regularization_losses).mean()
    
    # Total loss with Lagrange multiplier: L_λ = ||F_a|| + λ · L_traj
    total_loss = regularization_loss + lambda_current * trajectory_loss
    
    # === BACKWARD PASS ===
    # Backprop through residual network only
    # Physics prior is frozen (no grad updates for its parameters)
    total_loss.backward()
    optimizer.step()
    
    # === UPDATE LAGRANGE MULTIPLIER ===
    # Dual ascent: λ_{j+1} = λ_j + τ_2 · L_traj(θ_a)
    lambda_new = lambda_current + tau_2 * trajectory_loss.item()
    
    return {
        'loss_total': total_loss.item(),
        'loss_trajectory': trajectory_loss.item(),
        'loss_regularization': regularization_loss.item(),
        'lambda_new': lambda_new
    }


def train_phihp_epoch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Train using PhIHP (Physics-Informed Hybrid Physics) - one-step prediction error only.
    
    This approach trains on single-step prediction error, ignoring error compounding.
    Useful as a baseline or for initial training with more stable gradients.
    
    Placeholder for future implementation.
    """
    pass


def train_hybrid_epoch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Train using Hybrid approach - H-step trajectory prediction with loss on final state only.
    
    This approach predicts over the full H-step horizon but only computes loss on the final
    state. Balances between compounding error (APHYNITY) and single-step (PhIHP).
    
    Placeholder for future implementation.
    """
    pass


if __name__ == "__main__":
    """
    Test script for APHYNITY training objective.
    Loads minimal data, runs a few epochs with forward/backward passes,
    and prints all computed metrics.
    """
    
    # Import dependencies
    import sys
    import os
    sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')
    
    # Change to physics directory so relative paths work
    os.chdir('/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics')
    
    from fw_flightcontrol.physics.physics_prior import PhysicsPrior
    from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
    
    print("=" * 80)
    print("APHYNITY TRAINING OBJECTIVE - TEST SCRIPT")
    print("=" * 80)
    
    # Device
    device = torch.device('cpu')
    print(f"\nDevice: {device}")
    
    # Initialize models
    print("\nInitializing models...")
    physics_prior = PhysicsPrior()
    residual_network = PhysicsAugmented(state_dim=8, action_dim=3, hidden_dims=[64, 64])
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True
    )
    hybrid_model = hybrid_model.to(device)
    print(f"Physics Prior: {type(physics_prior).__name__}")
    print(f"Residual Network: {type(residual_network).__name__}")
    