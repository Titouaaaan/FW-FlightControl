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
    
    When training with normalized states:
    - Physics prior ALWAYS operates on raw physical units (denormalize state first)
    - Residual network operates on normalized states
    - Combined derivatives are in raw space (for ODE integration)
    
    Important: Create this once per epoch, not per step, to maintain
    consistent gradient tracking through the computation graph.
    """
    def __init__(self, hybrid_model: nn.Module, device: torch.device = torch.device('cpu'),
                 denorm_factors: torch.Tensor = None, min_bounds: torch.Tensor = None):
        super().__init__()
        self.model = hybrid_model
        self.device = device
        self.current_action = None
        # Denormalization factors for residual output: (max - min) / 2 per state
        self.denorm_factors = denorm_factors
        # Minimum bounds for state denormalization: needed to convert [-1,1] states back to raw
        self.min_bounds = min_bounds
    
    def set_action(self, action: torch.Tensor):
        self.current_action = action
    
    def _denormalize_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized state from [-1, 1] back to raw physical units.
        
        If normalized = 2*(raw - min)/(max - min) - 1, then:
        raw = (normalized + 1) * (max - min) / 2 + min
            = (normalized + 1) * denorm_factors + min_bounds
        """
        state_raw = (state_norm + 1.0) * self.denorm_factors + self.min_bounds
        return state_raw
    
    def _normalize_state(self, state_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw physical state back to normalized [-1, 1] space.
        
        If raw = (normalized + 1) * denorm_factors + min_bounds, then:
        normalized = (raw - min_bounds) / denorm_factors - 1
                   = 2 * (raw - min_bounds) / (max - min) - 1
        """
        state_norm = 2.0 * (state_raw - self.min_bounds) / (2.0 * self.denorm_factors) - 1.0
        return state_norm
    
    def forward(self, t: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
        """Compute state derivative: ds_raw/dt = F_p(s_raw, u) + F_a(s_norm(s_raw), u).
        
        CRITICAL: This wrapper works entirely in RAW PHYSICAL SPACE.
        - state_raw: raw physical state (what ODE integrator sends)
        - Returns: raw physical derivatives ds_raw/dt (what ODE integrator expects)
        
        Internally:
        - Physics prior operates on raw state → returns ds_raw/dt
        - Normalize raw state to normalized space for residual network
        - Residual network outputs normalized derivatives 
        - Denormalize residual derivatives to raw space
        - Combine both in raw space
        - Return raw derivatives for ODE integration
        """
        if self.current_action is None:
            raise RuntimeError("Action not set before forward pass")
        
        # When no normalization, state is raw, return raw derivatives
        if self.denorm_factors is None:
            hybrid_deriv = self.model(state_raw, self.current_action)
            return hybrid_deriv
        
        # Path when using normalization: receive raw, work with normalized, return raw
        # Normalize raw state to [-1, 1] for residual network
        state_norm = self._normalize_state(state_raw)
        
        # Physics prior operates on raw states (expects physical units)
        prior_deriv = self.model.physics_prior(state_raw, self.current_action)
        
        # Residual network operates on normalized states, outputs normalized derivatives
        residual_output = self.model.residual_network(state_norm, self.current_action)
        
        # Denormalize residual output: multiply by (max - min) / 2 to convert to raw space
        residual_output_denorm = residual_output * self.denorm_factors
        
        # Combine derivatives in raw space and return raw derivatives
        hybrid_deriv_raw = prior_deriv + residual_output_denorm
        
        return hybrid_deriv_raw


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
    ode_atol: float = 1e-5,
    dt: float = 0.01,
    denorm_factors: torch.Tensor = None,
    min_bounds: torch.Tensor = None,
    per_state_scales: torch.Tensor = None
) -> Dict:
    """
    Train residual network using APHYNITY (Augmented Physics with Newton's method).
    
    The APHYNITY approach trains on multi-step prediction error that compounds
    over H steps. This teaches the residual network to correct errors that grow
    when predictions are chained together.
    
    Mathematical formulation (from paper):
        θ_{j+1} = θ_j - τ₁∇[λⱼL_traj(θⱼ) + ‖F_a‖]
    
    where:
        - L = regularization + λ * trajectory_loss
        - τ₁ is gradient scaling (step size regularization for residual parameters)
        - λ is updated via dual ascent: λ_{j+1} = λ_j + τ₂ * L_traj
        - ∇ is gradient w.r.t. residual network parameters
    
    Args:
        hybrid_model: HybridDynamicsModel with physics_prior and residual_network
        trajectory_batch: Dictionary with trajectory data:
            - 'initial_states': (batch_size, 8) starting state s_0
            - 'actions': (batch_size, H, 3) sequence of actions
            - 'states': (batch_size, H, 8) ground truth trajectory s_1...s_H
        optimizer: Configured for residual_network parameters only
        lambda_current: Current λ value (Lagrange multiplier for dual ascent)
        tau_1: Gradient scaling factor (step size regularization). Applied as τ₁∇L before optimizer.step()
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
    hybrid_model = hybrid_model.to(device)
    optimizer.zero_grad()
    
    # ========================================================================
    # FORWARD PASS: Unroll H-step trajectory using specified integration method
    # ========================================================================
    predicted_states = []
    residual_norms = []
    
    current_state = initial_states
    
    # Use semi-implicit Euler if requested (custom method with gradient support)
    if ode_method == 'semi_implicit_euler':
        ode_module = HybridDynamicsODE(hybrid_model, device, denorm_factors=denorm_factors, min_bounds=min_bounds).to(device)
        # Manual semi-implicit Euler loop preserves gradient flow through unrolled loop
        for step in range(horizon):
            action = actions[:, step, :]  # (batch_size, 3)
            
            # For semi-implicit Euler, current_state starts in normalized space
            # Denormalize to raw for physics integration
            if denorm_factors is not None:
                current_state_raw = (current_state + 1.0) * denorm_factors + min_bounds
            else:
                current_state_raw = current_state
            
            # Compute residual output at this step for regularization term (only if model uses residual)
            if hybrid_model.with_residual:
                residual_output = hybrid_model.residual_network(current_state, action)
                
                # Denormalize residual to raw space before penalizing (if using normalization)
                # This ensures regularization penalty is consistent with trajectory loss (both in raw space)
                if denorm_factors is not None:
                    residual_output_raw = residual_output * denorm_factors
                    residual_norm = torch.norm(residual_output_raw, p=2, dim=1).mean()
                else:
                    residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
                residual_norms.append(residual_norm)
                
                # Check for NaN/Inf
                if torch.isnan(residual_norm) or torch.isinf(residual_norm):
                    raise ValueError(f"NaN/Inf in residual norm at step {step}")
            
            # Semi-implicit Euler step on RAW state: integrate in raw space
            # Use HybridDynamicsODE to ensure residual sees normalized state when normalization is enabled.
            ode_module.set_action(action)
            derivatives_raw = ode_module(None, current_state_raw)

            state_dim = current_state_raw.shape[-1]
            half_dim = state_dim // 2
            positions = current_state_raw[:, :half_dim]
            velocities = current_state_raw[:, half_dim:]

            dpos_dt = derivatives_raw[:, :half_dim]
            dvel_dt = derivatives_raw[:, half_dim:]

            velocities_new = velocities + dvel_dt * dt
            positions_new = positions + velocities_new * dt

            next_state_raw = torch.cat([positions_new, velocities_new], dim=-1)
            
            # Clamp raw state to valid physical bounds (prevent explosion)
            next_state_raw = next_state_raw.clamp(-1000.0, 1000.0)
            
            # Check for NaN after integration
            if torch.isnan(next_state_raw).any() or torch.isinf(next_state_raw).any():
                raise ValueError(f"NaN/Inf after semi-implicit Euler integration at step {step}")
            
            # Renormalize to normalized space for next iteration
            if denorm_factors is not None:
                next_state = 2.0 * (next_state_raw - min_bounds) / (2.0 * denorm_factors) - 1.0
            else:
                next_state = next_state_raw
            
            predicted_states.append(next_state)
            current_state = next_state
    
    else:
        # Use torchdiffeq for other ODE methods (RK4, dopri8, etc.)
        # ODE integration happens in RAW PHYSICAL SPACE
        ode_module = HybridDynamicsODE(hybrid_model, device, denorm_factors=denorm_factors, min_bounds=min_bounds).to(device)
        
        for step in range(horizon):
            action = actions[:, step, :]  # (batch_size, 3)
            
            # Compute residual output at this step for regularization term (only if model uses residual)
            if hybrid_model.with_residual:
                residual_output = hybrid_model.residual_network(current_state, action)
                
                # Denormalize residual to raw space before penalizing (if using normalization)
                # This ensures regularization penalty is consistent with trajectory loss (both in raw space)
                if denorm_factors is not None:
                    residual_output_raw = residual_output * denorm_factors
                    residual_norm = torch.norm(residual_output_raw, p=2, dim=1).mean()
                else:
                    residual_norm = torch.norm(residual_output, p=2, dim=1).mean()
                residual_norms.append(residual_norm)
                
                # Check for NaN/Inf
                if torch.isnan(residual_norm) or torch.isinf(residual_norm):
                    raise ValueError(f"NaN/Inf in residual norm at step {step}")
            
            # Denormalize current state to raw space for ODE integration
            if denorm_factors is not None:
                current_state_raw = (current_state + 1.0) * denorm_factors + min_bounds
            else:
                current_state_raw = current_state
            
            # Integrate one simulation step using torchdiffeq (in raw space)
            ode_module.set_action(action)
            t_eval = torch.tensor([0.0, dt], dtype=current_state_raw.dtype, device=device)
            
            solution = odeint(ode_module, current_state_raw, t_eval,
                             method=ode_method, rtol=ode_rtol, atol=ode_atol)
            next_state_raw = solution[-1]
            
            # Clamp raw state to valid physical bounds (prevent explosion)
            next_state_raw = next_state_raw.clamp(-1000.0, 1000.0)
            
            # Check for NaN after integration
            if torch.isnan(next_state_raw).any() or torch.isinf(next_state_raw).any():
                raise ValueError(f"NaN/Inf after {ode_method} integration at step {step}")
            
            # Renormalize to normalized space for next iteration
            if denorm_factors is not None:
                next_state = 2.0 * (next_state_raw - min_bounds) / (2.0 * denorm_factors) - 1.0
            else:
                next_state = next_state_raw
            
            predicted_states.append(next_state)
            current_state = next_state
    
    # Stack all predicted states
    predicted_trajectory = torch.stack(predicted_states, dim=1)  # (batch_size, H, 8)
    
    # ========================================================================
    # LOSS COMPUTATION
    # ========================================================================
    
    # Denormalize predictions and ground truth to raw space if using normalization
    # This ensures loss is computed in raw (physical) space for meaningful values
    if denorm_factors is not None:
        # predicted_trajectory and ground_truth_states are in normalized space
        # Denormalize: raw = (normalized + 1) * denorm_factors + min_bounds
        predicted_trajectory_raw = (predicted_trajectory + 1.0) * denorm_factors + min_bounds
        ground_truth_states_raw = (ground_truth_states + 1.0) * denorm_factors + min_bounds
    else:
        predicted_trajectory_raw = predicted_trajectory
        ground_truth_states_raw = ground_truth_states
    
    # Trajectory loss: multi-step prediction error across all steps (in raw space)
    # We compute the L2 distance for each (sample, step) pair
    prediction_error = predicted_trajectory_raw - ground_truth_states_raw  # (batch, H, 8)
    if per_state_scales is not None:
        # Inverse scaling: multiply by 1/per_state_scales so small-scale states (angular rates)
        # get amplified gradients and large-scale states (airspeed) get reduced gradients
        prediction_error = prediction_error / (per_state_scales ** 2)
    trajectory_loss = torch.norm(prediction_error, p=2, dim=2).mean()  # Average L2 norm
    
    # Check for NaN in trajectory loss
    if torch.isnan(trajectory_loss):
        print(f"NaN detected in trajectory_loss")
        print(f"  prediction_error range: [{prediction_error.min()}, {prediction_error.max()}]")
        print(f"  trajectory_loss: {trajectory_loss}")
        raise ValueError("NaN in trajectory loss")
    
    # Regularization loss: keep residual magnitudes small (only if model uses residual)
    # This prevents the network from learning large corrections that don't generalize
    if residual_norms:
        regularization_loss = torch.stack(residual_norms).mean()
    else:
        # No residual component; regularization is zero
        regularization_loss = torch.tensor(0.0, device=state.device, dtype=state.dtype)
    
    # Check for NaN in regularization loss
    if torch.isnan(regularization_loss):
        print(f"NaN detected in regularization_loss")
        print(f"  residual_norms: {torch.stack(residual_norms)}")
        raise ValueError("NaN in regularization loss")
    
    # Combined APHYNITY loss: regularization + λ * trajectory_loss
    # Note: τ_1 is applied to gradients, not the loss itself (see APHYNITY paper)
    total_loss = regularization_loss + lambda_current * trajectory_loss
    
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
    
    # Compute gradient statistics before clipping
    grad_norms_before = []
    for param in hybrid_model.residual_network.parameters():
        if param.grad is not None:
            grad_norms_before.append(torch.norm(param.grad).item())
    
    grad_norm_before_clipping = sum(grad_norms_before) / len(grad_norms_before) if grad_norms_before else 0.0
    grad_max_before_clipping = max(grad_norms_before) if grad_norms_before else 0.0
    
    # Clip gradients for stability
    grad_norm_clipped = torch.nn.utils.clip_grad_norm_(hybrid_model.residual_network.parameters(), 
                                                       max_norm=1.0)
    
    # Compute gradient statistics after clipping
    grad_norms_after = []
    for param in hybrid_model.residual_network.parameters():
        if param.grad is not None:
            grad_norms_after.append(torch.norm(param.grad).item())
    
    grad_norm_after_clipping = sum(grad_norms_after) / len(grad_norms_after) if grad_norms_after else 0.0
    grad_max_after_clipping = max(grad_norms_after) if grad_norms_after else 0.0
    
    # Update residual network parameters
    # Note: tau_1 is handled by Adam optimizer as learning rate, NOT applied here
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
        'lambda_new': lambda_new,
        'grad_norm_before_clipping': grad_norm_before_clipping,
        'grad_max_before_clipping': grad_max_before_clipping,
        'grad_norm_after_clipping': grad_norm_after_clipping,
        'grad_max_after_clipping': grad_max_after_clipping,
        'grad_norm_clipped': grad_norm_clipped.item() if isinstance(grad_norm_clipped, torch.Tensor) else grad_norm_clipped
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

    