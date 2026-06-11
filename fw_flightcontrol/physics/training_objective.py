import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchdiffeq import odeint
from typing import Dict
from .utils import normalize_state_torch, denormalize_state_torch


class HybridDynamicsODE(nn.Module):
    """Wrapper for hybrid dynamics to work with torchdiffeq.odeint.

    Always operates in RAW PHYSICAL SPACE. On the first call after arm_capture(),
    stores the residual norm so the training loop can use it for regularization
    without an extra forward pass (captures from the k1 evaluation in RK4).

    Norm parameters are read from hybrid_model.norm_scale / norm_offset.
    """
    def __init__(self, hybrid_model: nn.Module, device: torch.device,
                 residual_clamp: float = None):
        super().__init__()
        self.model          = hybrid_model
        self.device         = device
        self.current_action = None
        self.residual_clamp = residual_clamp
        self._capture_next  = False
        self.captured_residual_norm = None

    def set_action(self, action: torch.Tensor):
        self.current_action = action

    def arm_capture(self):
        """Arm to capture residual norm on next forward call (k1 for RK4)."""
        self._capture_next = True
        self.captured_residual_norm = None

    def forward(self, t: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
        """Compute ds_raw/dt = F_p(s_raw, u) + F_a(s_norm, u) * std."""
        if self.current_action is None:
            raise RuntimeError("Action not set before forward pass")

        norm_scale  = self.model.norm_scale
        norm_offset = self.model.norm_offset

        dx_dt = torch.zeros_like(state_raw)

        if self.model.with_prior:
            dx_dt = self.model.physics_prior(state_raw, self.current_action)

        if self.model.with_residual:
            state_norm     = (state_raw - norm_offset) / norm_scale
            residual_output = self.model.residual_network(state_norm, self.current_action)
            if self.residual_clamp is not None:
                residual_output = residual_output.clamp(-self.residual_clamp, self.residual_clamp)
            if self._capture_next:
                # Norm in raw space: multiply normalized residual by std
                self.captured_residual_norm = torch.norm(
                    residual_output * norm_scale, p=2, dim=1
                ).mean()
                self._capture_next = False
            dx_dt = dx_dt + residual_output * norm_scale
        elif self._capture_next:
            self._capture_next = False

        return dx_dt


def train_aphynity_epoch(
    hybrid_model: nn.Module,
    trajectory_batch: Dict,
    optimizer: Optimizer,
    lambda_current: float,
    tau_2: float,
    grad_clip_norm: float = 1.0,
    device: torch.device = torch.device('cpu'),
    ode_method: str = 'dopri5',
    ode_rtol: float = 1e-4,
    ode_atol: float = 1e-5,
    dt: float = 0.01,
) -> Dict:

    initial_states      = trajectory_batch['initial_states'].to(device)
    actions             = trajectory_batch['actions'].to(device)
    ground_truth_states = trajectory_batch['states'].to(device)

    horizon = actions.shape[1]
    optimizer.zero_grad()

    predicted_states = []
    residual_norms   = []
    current_state    = initial_states

    ode_module = HybridDynamicsODE(hybrid_model, device).to(device)
    t_eval = torch.tensor([0.0, dt], dtype=initial_states.dtype, device=device)

    if ode_method == 'semi_implicit_euler':
        for step in range(horizon):
            action = actions[:, step, :]
            current_state_raw = denormalize_state_torch(
                current_state, hybrid_model.norm_scale, hybrid_model.norm_offset
            )
            ode_module.set_action(action)
            ode_module.arm_capture()
            derivatives_raw = ode_module(None, current_state_raw)

            if hybrid_model.with_residual and ode_module.captured_residual_norm is not None:
                residual_norms.append(ode_module.captured_residual_norm)

            state_dim  = current_state_raw.shape[-1]
            half_dim   = state_dim // 2
            vel_new    = current_state_raw[:, half_dim:] + derivatives_raw[:, half_dim:] * dt
            pos_new    = current_state_raw[:, :half_dim] + vel_new * dt
            next_state_raw = torch.cat([pos_new, vel_new], dim=-1).clamp(-1000.0, 1000.0)

            next_state = normalize_state_torch(
                next_state_raw, hybrid_model.norm_scale, hybrid_model.norm_offset
            )
            predicted_states.append(next_state)
            current_state = next_state

    else:
        for step in range(horizon):
            action = actions[:, step, :]
            current_state_raw = denormalize_state_torch(
                current_state, hybrid_model.norm_scale, hybrid_model.norm_offset
            )
            ode_module.set_action(action)
            ode_module.arm_capture()
            solution = odeint(ode_module, current_state_raw, t_eval,
                              method=ode_method, rtol=ode_rtol, atol=ode_atol)

            if hybrid_model.with_residual and ode_module.captured_residual_norm is not None:
                residual_norms.append(ode_module.captured_residual_norm)

            next_state_raw = solution[-1].clamp(-1000.0, 1000.0)
            next_state = normalize_state_torch(
                next_state_raw, hybrid_model.norm_scale, hybrid_model.norm_offset
            )
            predicted_states.append(next_state)
            current_state = next_state

    predicted_trajectory = torch.stack(predicted_states, dim=1)  # (B, H, 8)

    # Loss is computed in normalized space (equivalent to std-weighted raw loss)
    trajectory_loss = torch.norm(predicted_trajectory - ground_truth_states, p=2, dim=2).mean()

    if torch.isnan(trajectory_loss):
        raise ValueError("NaN in trajectory loss")

    regularization_loss = (
        torch.stack(residual_norms).mean() if residual_norms
        else torch.tensor(0.0, device=device, dtype=initial_states.dtype)
    )

    if torch.isnan(regularization_loss):
        raise ValueError("NaN in regularization loss")

    total_loss = regularization_loss + lambda_current * trajectory_loss

    if torch.isnan(total_loss):
        raise ValueError("NaN in total loss")

    total_loss.backward()

    grad_norm_before = torch.nn.utils.clip_grad_norm_(
        hybrid_model.residual_network.parameters(), max_norm=grad_clip_norm
    ).item()

    params_with_grad = [p for p in hybrid_model.residual_network.parameters() if p.grad is not None]
    grad_norm_after = (
        torch.stack([p.grad.detach().norm() for p in params_with_grad]).norm().item()
        if params_with_grad else 0.0
    )

    optimizer.step()

    # Dual ascent: λ_{new} = λ + τ₂ * L_traj
    lambda_new = lambda_current + tau_2 * trajectory_loss.item()

    return {
        'loss_total':                total_loss.item(),
        'loss_trajectory':           trajectory_loss.item(),
        'loss_regularization':       regularization_loss.item(),
        'lambda_new':                lambda_new,
        'grad_norm_before_clipping': grad_norm_before,
        'grad_norm_after_clipping':  grad_norm_after,
    }
