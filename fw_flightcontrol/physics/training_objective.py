"""
Training objectives for hybrid physics-augmented world model.

This module implements APHYNITY-style training, which optimizes the residual
network in the context of multi-step prediction. The key idea is that we want
to minimize compounding errors over H steps, not just single-step errors.

The APHYNITY loss has two components:
1. Regularization term: τ_1 * ||F_a(s,u)||^2 - keeps residuals small
2. Trajectory error: λ * Σ||s_pred - s_true||^2 - minimizes multi-step errors

The parameter λ is updated via dual ascent, automatically balancing the two terms.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchdiffeq import odeint
from typing import Dict


class HybridDynamicsODE(nn.Module):
    """Wrapper for hybrid dynamics to work with torchdiffeq.odeint.
    
    This module wraps the hybrid model (physics prior + residual network)
    to provide proper gradient flow through ODE integration.
    
    Important: Create this once per epoch, not per step, to maintain
    consistent gradient tracking through the computation graph.
    """
    def __init__(self, hybrid_model: nn.Module, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.model = hybrid_model
        self.device = device
        self.current_action = None
    
    def set_action(self, action: torch.Tensor):
        """Set the action for the next integration step."""
        self.current_action = action
    
    def forward(self, t: torch.Tensor, state_t: torch.Tensor) -> torch.Tensor:
        """Compute state derivative: ds/dt = F_p(s, u) + F_a(s, u)."""
        if self.current_action is None:
            raise RuntimeError("Action not set before forward pass")
        return self.model(state_t, self.current_action)


def train_aphynity_epoch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    lambda_current: float,
    tau_1: float,
    tau_2: float,
    device: torch.device = torch.device('cpu'),
    ode_method: str = 'dopri5',
    ode_rtol: float = 1e-4,
    ode_atol: float = 1e-5
) -> Dict:
    """
    Train residual network using APHYNITY (Augmented Physics with Newton's method).
    
    The APHYNITY approach trains on multi-step prediction error that compounds
    over H steps. This teaches the residual network to correct errors that grow
    when predictions are chained together.
    
    Mathematical formulation:
        L_λ(θ_a) = (1/H) Σ_k ||F_a(s_k, u_k)||^2 + λ * (1/NH) Σ_i,k ||ŝ_k^(i) - s_k^(i)||^2
    
    where:
        - First term regularizes residual magnitude (smoothness constraint)
        - Second term minimizes trajectory prediction error over H steps
        - λ controls the trade-off (updated via dual ascent)
    
    Args:
        hybrid_model: HybridDynamicsModel with physics_prior and residual_network
        trajectory_batch: Dictionary with trajectory data:
            - 'initial_states': (batch_size, 8) starting state s_0
            - 'actions': (batch_size, H, 3) sequence of actions
            - 'states': (batch_size, H, 8) ground truth trajectory s_1...s_H
        optimizer: Configured for residual_network parameters only
        lambda_current: Current λ value (Lagrange multiplier for dual ascent)
        tau_1: Regularization coefficient weight on ||F_a||^2 term
        tau_2: Step size for λ update
        device: CPU or CUDA
    
    Returns:
        Dictionary with metrics for this epoch
    """
    
    # Load batch data
    initial_states = trajectory_batch['initial_states'].to(device)  # (batch_size, 8)
    actions = trajectory_batch['actions'].to(device)                # (batch_size, H, 3)
    ground_truth_states = trajectory_batch['states'].to(device)     # (batch_size, H, 8)
    
    batch_size = initial_states.shape[0]
    horizon = actions.shape[1]
    
    # Ensure model is on correct device (handles GPU/CPU)
    hybrid_model = hybrid_model.to(device)
    
    # Clear gradients from previous iteration
    optimizer.zero_grad()
    
    # ========================================================================
    # FORWARD PASS: Unroll H-step trajectory using RK4 integration
    # ========================================================================
    predicted_states = []
    residual_norms = []
    
    # Create ODE wrapper once - critical for gradient tracking
    ode_module = HybridDynamicsODE(hybrid_model, device).to(device)
    
    current_state = initial_states
    
    for step in range(horizon):
        action = actions[:, step, :]  # (batch_size, 3)
        
        # DEBUG: Check input state ranges
        state_max = torch.abs(current_state).max().item()
        action_max = torch.abs(action).max().item()
        
        # Compute residual output at this step for regularization term
        # These are the corrections F_a(s_k, u_k) that we want to keep small
        residual_output = hybrid_model.residual_network(current_state, action)
        residual_max = torch.abs(residual_output).max().item()
        residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
        residual_norms.append(residual_norm)
        
        # DEBUG: Check physics prior output
        physics_output = hybrid_model.physics_prior(current_state, action)
        physics_max = torch.abs(physics_output).max().item()
        
        # Check for NaN/Inf in inputs or outputs
        if torch.isnan(residual_norm) or torch.isinf(residual_norm):
            print(f"\n{'='*80}")
            print(f"NaN/Inf detected at step {step}")
            print(f"  Input state range: [{current_state.min():.4e}, {current_state.max():.4e}]")
            print(f"  Input action range: [{action.min():.4e}, {action.max():.4e}]")
            print(f"  Physics output range: [{physics_output.min():.4e}, {physics_output.max():.4e}]")
            print(f"  Residual output range: [{residual_output.min():.4e}, {residual_output.max():.4e}]")
            print(f"  Physics max magnitude: {physics_max:.4e}")
            print(f"  Residual max magnitude: {residual_max:.4e}")
            print(f"  Residual norm: {residual_norm}")
            print(f"{'='*80}\n")
            raise ValueError(f"NaN/Inf in residual norm at step {step}")
        
        # Integrate one simulation step: s_{k+1} = s_k + ∫[s_k, 0.01] (F_p + F_a) dt
        # Set the action for this step
        ode_module.set_action(action)
        
        # Time points for integration
        t_eval = torch.tensor([0.0, 0.01], dtype=current_state.dtype, device=device)
        
        # Integrate using RK4 method with the wrapper module
        solution = odeint(ode_module, current_state, t_eval,
                         method=ode_method, rtol=ode_rtol, atol=ode_atol)
        next_state = solution[-1]  # Extract final state at t=0.01
        
        # DEBUG: Check integration output before clamping
        next_state_max = torch.abs(next_state).max().item()
        
        # Clamp state values to prevent numerical explosion
        # Expected ranges based on flight dynamics (angles in rad ~[-pi, pi], 
        # angular rates ~[-10, 10], airspeed ~[0, 50] m/s)
        next_state = next_state.clamp(-100.0, 100.0)
        
        # Check for NaN in integrated state
        if torch.isnan(next_state).any() or torch.isinf(next_state).any():
            print(f"\n{'='*80}")
            print(f"NaN/Inf detected after integration at step {step}")
            print(f"  Pre-clamp state range: [{solution[-1].min():.4e}, {solution[-1].max():.4e}]")
            print(f"  Post-clamp state range: [{next_state.min():.4e}, {next_state.max():.4e}]")
            print(f"  Pre-clamp max magnitude: {next_state_max:.4e}")
            print(f"{'='*80}\n")
            raise ValueError(f"NaN/Inf in integration at step {step}")
        
        predicted_states.append(next_state)
        current_state = next_state
    
    # Stack all predicted states
    predicted_trajectory = torch.stack(predicted_states, dim=1)  # (batch_size, H, 8)
    
    # ========================================================================
    # LOSS COMPUTATION
    # ========================================================================
    
    # Trajectory loss: multi-step prediction error across all steps
    # We compute the L2 distance for each (sample, step) pair
    prediction_error = predicted_trajectory - ground_truth_states  # (batch, H, 8)
    trajectory_loss = torch.norm(prediction_error, p=2, dim=2).mean()  # Average L2 norm
    
    # Check for NaN in trajectory loss
    if torch.isnan(trajectory_loss):
        print(f"NaN detected in trajectory_loss")
        print(f"  prediction_error range: [{prediction_error.min()}, {prediction_error.max()}]")
        print(f"  trajectory_loss: {trajectory_loss}")
        raise ValueError("NaN in trajectory loss")
    
    # Regularization loss: keep residual magnitudes small
    # This prevents the network from learning large corrections that don't generalize
    regularization_loss = torch.stack(residual_norms).mean()
    
    # Check for NaN in regularization loss
    if torch.isnan(regularization_loss):
        print(f"NaN detected in regularization_loss")
        print(f"  residual_norms: {torch.stack(residual_norms)}")
        raise ValueError("NaN in regularization loss")
    
    # Combined APHYNITY loss with explicit regularization weight and Lagrange multiplier
    # L_λ = τ_1 * regularization + λ * trajectory_loss
    # τ_1 ensures regularization isn't overwhelmed by trajectory loss
    total_loss = tau_1 * regularization_loss + lambda_current * trajectory_loss
    
    # Check for NaN in total loss
    if torch.isnan(total_loss):
        print(f"NaN detected in total_loss")
        print(f"  regularization_loss: {regularization_loss}")
        print(f"  lambda_current: {lambda_current}")
        print(f"  trajectory_loss: {trajectory_loss}")
        raise ValueError("NaN in total loss")
    
    # ========================================================================
    # BACKWARD PASS
    # ========================================================================
    # Backpropagate through the entire unrolled trajectory
    # PyTorch's autograd automatically handles this through torchdiffeq
    total_loss.backward()
    
    # Check for NaN in gradients
    for name, param in hybrid_model.residual_network.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradients of {name}")
            print(f"  grad range: [{param.grad.min()}, {param.grad.max()}]")
            raise ValueError(f"NaN gradient in {name}")
    
    # Clip gradients for stability
    torch.nn.utils.clip_grad_norm_(hybrid_model.residual_network.parameters(), 
                                   max_norm=1.0)
    
    # Update residual network parameters
    optimizer.step()
    
    # ========================================================================
    # DUAL ASCENT: Update Lagrange multiplier
    # ========================================================================
    # λ_{new} = λ + τ_2 * trajectory_loss
    # This increases λ when trajectory loss is high (emphasizing trajectory matching)
    # and keeps it stable when trajectory loss is low (emphasizing regularization)
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
    PhIHP training: one-step prediction error only (baseline).
    
    This approach ignores error compounding and optimizes single-step accuracy.
    Useful as a simpler baseline for comparison.
    
    TODO: Implement this approach
    """
    pass


def train_hybrid_epoch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Hybrid training: H-step prediction with loss only on final state.
    
    Balances between APHYNITY (all steps) and PhIHP (single step by doing
    multi-step prediction but only measuring error at the end.
    
    TODO: Implement this approach
    """
    pass

    