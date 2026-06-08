"""Training objectives for hybrid physics-augmented world model."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchdiffeq import odeint
from typing import Dict, Optional

from .utils import normalize_state_torch, denormalize_state_torch


class HybridDynamicsODE(nn.Module):
    """Wraps the hybrid model for torchdiffeq.odeint.

    Operates entirely in raw physical space. Residual network receives
    normalized state when norm_type is set.

    Invariant: ds_raw/dt = F_p(s_raw, u) + F_a(normalize(s_raw), u) * std
    """
    def __init__(self, hybrid_model, device=torch.device('cpu'),
                 denorm_factors=None, min_bounds=None,
                 norm_type=None, residual_clamp=None):
        super().__init__()
        self.model = hybrid_model
        self.device = device
        self.current_action = None
        self.current_prev_action = None
        self.denorm_factors = denorm_factors
        self.min_bounds = min_bounds
        self.norm_type = norm_type
        self.residual_clamp = residual_clamp

    def set_action(self, action: torch.Tensor):
        self.current_action = action

    def set_prev_action(self, prev_action: torch.Tensor):
        self.current_prev_action = prev_action

    def forward(self, t: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
        if self.current_action is None:
            raise RuntimeError("Action not set before forward pass")

        state_raw = state_raw.clamp(-1000.0, 1000.0)

        dx_dt = torch.zeros_like(state_raw)
        if self.model.with_prior:
            dx_dt = self.model.physics_prior(state_raw, self.current_action)

        if self.model.with_residual:
            if self.denorm_factors is not None:
                state_norm = normalize_state_torch(
                    state_raw, self.denorm_factors, self.min_bounds, self.norm_type
                )
                residual_output = self.model.residual_network(
                    state_norm, self.current_action, self.current_prev_action
                )
                if self.residual_clamp is not None:
                    residual_output = residual_output.clamp(-self.residual_clamp, self.residual_clamp)
                dx_dt = dx_dt + residual_output * self.denorm_factors
            else:
                residual_output = self.model.residual_network(
                    state_raw, self.current_action, self.current_prev_action
                )
                if self.residual_clamp is not None:
                    residual_output = residual_output.clamp(-self.residual_clamp, self.residual_clamp)
                dx_dt = dx_dt + residual_output if self.model.with_prior else residual_output

        return dx_dt


def rollout(
    ode_module: HybridDynamicsODE,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    t_eval: torch.Tensor,
    ode_method: str = 'rk4',
    ode_rtol: float = 1e-4,
    ode_atol: float = 1e-5,
    denorm_factors: torch.Tensor = None,
    min_bounds: torch.Tensor = None,
    norm_type: Optional[str] = None,
    prev_actions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """H-step rollout using torchdiffeq. Returns predicted trajectory in same
    space as initial_states (normalized when norm_type is set)."""
    horizon = actions.shape[1]
    predicted_states = []
    current_state = initial_states

    for step in range(horizon):
        action = actions[:, step, :]
        prev_action = prev_actions[:, step, :] if prev_actions is not None else None

        if denorm_factors is not None:
            current_state_raw = denormalize_state_torch(
                current_state, denorm_factors, min_bounds, norm_type
            )
        else:
            current_state_raw = current_state

        ode_module.set_action(action)
        ode_module.set_prev_action(prev_action)
        solution = odeint(ode_module, current_state_raw, t_eval,
                          method=ode_method, rtol=ode_rtol, atol=ode_atol)
        next_state_raw = solution[-1].clamp(-1000.0, 1000.0)

        if denorm_factors is not None:
            next_state = normalize_state_torch(next_state_raw, denorm_factors, min_bounds, norm_type)
        else:
            next_state = next_state_raw

        predicted_states.append(next_state)
        current_state = next_state

    return torch.stack(predicted_states, dim=1)  # (batch, H, state_dim)


def train_aphynity_batch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    lambda_current: float,
    tau_2: float,
    grad_clip_norm: float = 1.0,
    device: torch.device = torch.device('cpu'),
    ode_method: str = 'rk4',
    ode_rtol: float = 1e-4,
    ode_atol: float = 1e-5,
    dt: float = 0.01,
    denorm_factors: torch.Tensor = None,
    min_bounds: torch.Tensor = None,
    norm_type: Optional[str] = None,
) -> Dict:
    """One APHYNITY batch step.

    Regularization is computed on ground-truth states (not captured during
    rollout). Loss = ((F_a(s_gt) * std)^2).mean() + lambda * MSE(pred, gt).
    """
    initial_states = trajectory_batch['initial_states'].to(device)
    actions        = trajectory_batch['actions'].to(device)
    gt_states      = trajectory_batch['states'].to(device)
    prev_actions   = trajectory_batch.get('prev_actions')
    if prev_actions is not None:
        prev_actions = prev_actions.to(device)

    optimizer.zero_grad()

    ode_module = HybridDynamicsODE(
        hybrid_model, device,
        denorm_factors=denorm_factors, min_bounds=min_bounds, norm_type=norm_type,
    ).to(device)
    t_eval = torch.tensor([0.0, dt], dtype=initial_states.dtype, device=device)

    # Regularization on ground-truth initial states
    if hybrid_model.with_residual:
        residual_reg = hybrid_model.residual_network(
            initial_states, actions[:, 0, :],
            prev_actions[:, 0, :] if prev_actions is not None else None,
        )
        if denorm_factors is not None:
            regularization = ((residual_reg * denorm_factors) ** 2).mean()
        else:
            regularization = (residual_reg ** 2).mean()
    else:
        regularization = torch.tensor(0.0, device=device, dtype=initial_states.dtype)

    # Multi-step rollout
    predicted_trajectory = rollout(
        ode_module, initial_states, actions, t_eval,
        ode_method=ode_method, ode_rtol=ode_rtol, ode_atol=ode_atol,
        denorm_factors=denorm_factors, min_bounds=min_bounds, norm_type=norm_type,
        prev_actions=prev_actions,
    )

    # Both predicted and gt are in the same space (normalized or raw) — MSE directly
    trajectory_loss = nn.functional.mse_loss(predicted_trajectory, gt_states)

    if not torch.isfinite(trajectory_loss):
        raise ValueError(f"Non-finite trajectory loss: {trajectory_loss.item()}")
    if not torch.isfinite(regularization):
        raise ValueError(f"Non-finite regularization: {regularization.item()}")

    total_loss = regularization + lambda_current * trajectory_loss
    total_loss.backward()

    has_bad_grad = any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in hybrid_model.residual_network.parameters()
    )
    if has_bad_grad:
        optimizer.zero_grad()
        raise ValueError("NaN/Inf in gradients — weight update skipped")

    grad_norm = torch.nn.utils.clip_grad_norm_(
        hybrid_model.residual_network.parameters(), max_norm=grad_clip_norm
    ).item()

    optimizer.step()

    lambda_new = lambda_current + tau_2 * trajectory_loss.item()

    return {
        'loss_total':          total_loss.item(),
        'loss_trajectory':     trajectory_loss.item(),
        'loss_regularization': regularization.item(),
        'lambda_new':          lambda_new,
        'grad_norm':           grad_norm,
    }
