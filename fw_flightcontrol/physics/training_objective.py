import torch
from torchdiffeq import odeint
from typing import Optional

from .utils import normalize_state_torch, denormalize_state_torch


class HybridDynamicsODE(torch.nn.Module):
    def __init__(self, hybrid_model, device, denorm_factors, min_bounds, norm_type):
        super().__init__()
        self.model = hybrid_model
        self.device = device
        self.denorm_factors = denorm_factors
        self.min_bounds = min_bounds
        self.norm_type = norm_type
        self.current_action: Optional[torch.Tensor] = None
        self.current_prev_action: Optional[torch.Tensor] = None

    def set_action(self, action: torch.Tensor):
        self.current_action = action

    def set_prev_action(self, prev_action: Optional[torch.Tensor]):
        self.current_prev_action = prev_action

    def forward(self, _t: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
        state_raw = state_raw.clamp(-1000.0, 1000.0)
        dx_dt = torch.zeros_like(state_raw)
        if self.model.with_prior:
            dx_dt = self.model.physics_prior(state_raw, self.current_action)
        if self.model.with_residual:
            state_norm = normalize_state_torch(state_raw, self.denorm_factors, self.min_bounds, self.norm_type)
            residual = self.model.residual_network(state_norm, self.current_action, self.current_prev_action)
            dx_dt = dx_dt + residual * self.denorm_factors
        return dx_dt


def rollout(ode_module, t_eval, initial_states, actions, prev_actions, horizon, ode_method, ode_rtol, ode_atol):
    """Returns predicted_trajectory (B, H, state_dim) in normalized space."""
    predicted_states = []
    current_state = initial_states
    for step in range(horizon):
        current_state_raw = denormalize_state_torch(
            current_state, ode_module.denorm_factors, ode_module.min_bounds, ode_module.norm_type
        )
        ode_module.set_action(actions[:, step, :])
        ode_module.set_prev_action(prev_actions[:, step, :] if prev_actions is not None else None)
        solution = odeint(ode_module, current_state_raw, t_eval, method=ode_method, rtol=ode_rtol, atol=ode_atol)
        next_state_raw = solution[-1].clamp(-1000.0, 1000.0)
        current_state = normalize_state_torch(
            next_state_raw, ode_module.denorm_factors, ode_module.min_bounds, ode_module.norm_type
        ).clamp(-10.0, 10.0)
        predicted_states.append(current_state)
    return torch.stack(predicted_states, dim=1)


def train_aphynity_batch(ode_module, t_eval, trajectory_batch, optimizer, lambda_current,
                         grad_clip_norm, device, ode_method, ode_rtol, ode_atol):
    """L = ||F_a||² + λ * MSE(pred, target)  — matches official APHYNITY."""
    initial_states = trajectory_batch['initial_states'].to(device)
    actions        = trajectory_batch['actions'].to(device)
    ground_truth   = trajectory_batch['states'].to(device)
    prev_actions   = trajectory_batch.get('prev_actions')
    if prev_actions is not None:
        prev_actions = prev_actions.to(device)

    optimizer.zero_grad()

    predicted_trajectory = rollout(
        ode_module, t_eval, initial_states, actions, prev_actions,
        actions.shape[1], ode_method, ode_rtol, ode_atol,
    )

    trajectory_loss = ((predicted_trajectory - ground_truth) ** 2).mean()
    if not torch.isfinite(trajectory_loss):
        raise ValueError(f"Non-finite trajectory loss: {trajectory_loss.item()}")

    # Regularizer evaluated on ground truth states — matches official APHYNITY
    B, H, S = ground_truth.shape
    residual_gt = ode_module.model.residual_network(
        ground_truth.view(B * H, S),
        actions.view(B * H, -1),
        prev_actions.view(B * H, -1) if prev_actions is not None else None,
    )
    regularization_loss = ((residual_gt * ode_module.denorm_factors) ** 2).sum(dim=1).mean()
    if not torch.isfinite(regularization_loss):
        raise ValueError(f"Non-finite regularization loss: {regularization_loss.item()}")

    total_loss = regularization_loss + lambda_current * trajectory_loss
    total_loss.backward()

    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for p in ode_module.model.residual_network.parameters()):
        optimizer.zero_grad()
        raise ValueError("NaN/Inf in gradients — weight update skipped")

    grad_norm = torch.nn.utils.clip_grad_norm_(
        ode_module.model.residual_network.parameters(), max_norm=grad_clip_norm
    ).item()
    optimizer.step()

    return {
        'loss_total':          total_loss.item(),
        'loss_trajectory':     trajectory_loss.item(),
        'loss_regularization': regularization_loss.item(),
        'grad_norm':           grad_norm,
    }
