#!/usr/bin/env python3
"""
Residual learning network and hybrid dynamics model for aircraft dynamics.

The key insight: the physics prior captures the basic aerodynamic equations,
but real aircraft have modeling errors (unmodeled effects, coefficient uncertainties).
We use a learned residual network F_a to correct these errors:

    ds/dt = F_p(s, u) + F_a(s, u)

where F_p is the frozen physics prior and F_a is the learnable residual network.
This hybrid approach combines model-based constraints with learned corrections.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class PhysicsAugmented(nn.Module):
    """
    Learned residual network that corrects physics prior predictions.

    Input: current state s and action u
    Output: residual corrections Δ(ds/dt) to add to physics prior

    We use a simple MLP architecture for efficiency and stability.
    The network is initialized with small weights to ensure the residuals
    start near zero, letting the physics prior dominate early training.
    """

    def __init__(self,
                 state_dim: int = 8,
                 action_dim: int = 3,
                 hidden_dims: list = None,
                 activation: str = 'relu',
                 use_batch_norm: bool = False,
                 prev_action_dim: int = 0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prev_action_dim = prev_action_dim
        self.output_dim = state_dim  # Output residuals have same dimension as state derivatives

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.use_batch_norm = use_batch_norm

        # Build the MLP: [state, action, (prev_action)] -> hidden layers -> residual corrections
        layers = []
        input_dim = state_dim + action_dim + prev_action_dim

        # Hidden layers with optional batch norm
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim

        # Output layer: no activation (residuals are unbounded)
        layers.append(nn.Linear(input_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights to small values for stable residual learning
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                prev_action: torch.Tensor = None) -> torch.Tensor:
        parts = [state, action]
        if self.prev_action_dim > 0 and prev_action is not None:
            parts.append(prev_action)
        x = torch.cat(parts, dim=-1)
        return self.network(x)


class HybridDynamicsModel(nn.Module):
    """
    Combines physics prior and learned residual network into a unified dynamics model.

    This allows for flexible ablation studies:
    - Physics only: with_prior=True, with_residual=False
    - Learning only: with_prior=False, with_residual=True
    - Hybrid: with_prior=True, with_residual=True (full model)

    The combined dynamics are: ds/dt = F_p(s,u) + F_a(s,u)

    Supports multiple ODE integration methods for accuracy vs speed tradeoff.
    """

    def __init__(self,
                 physics_prior,
                 residual_network,
                 with_prior: bool = True,
                 with_residual: bool = True,
                 integration_method: str = 'rk4'):
        super().__init__()

        self.physics_prior = physics_prior
        self.residual_network = residual_network
        self.with_prior = with_prior
        self.with_residual = with_residual
        self.integration_method = integration_method

        if integration_method not in ['rk4', 'dopri8', 'semi_implicit_euler']:
            raise ValueError(f"integration_method must be 'rk4', 'dopri8', or 'semi_implicit_euler', got '{integration_method}'")

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dx_dt_combined = torch.zeros_like(state)

        if self.with_prior:
            dx_dt_physics = self.physics_prior(state, action)
            dx_dt_combined = dx_dt_physics

        if self.with_residual:
            residuals = self.residual_network(state, action)
            if self.with_prior:
                dx_dt_combined = dx_dt_combined + residuals
            else:
                dx_dt_combined = residuals

        return dx_dt_combined

    def integrate_rk4(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        def ode_dynamics(t, state_t):
            return self(state_t, action)
        t_eval = torch.tensor([0.0, 0.01], dtype=state.dtype, device=state.device)
        trajectory = odeint(ode_dynamics, state, t_eval, method='rk4')
        return trajectory[-1]

    def integrate_dop853(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        def ode_dynamics(t, state_t):
            return self(state_t, action)
        t_eval = torch.tensor([0.0, 0.01], dtype=state.dtype, device=state.device)
        trajectory = odeint(ode_dynamics, state, t_eval, method='dopri8', rtol=1e-8, atol=1e-9)
        return trajectory[-1]

    def integrate_semi_implicit_euler(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        derivatives = self(state, action)
        state_dim = state.shape[-1]
        half_dim = state_dim // 2
        positions = state[:, :half_dim]
        velocities = state[:, half_dim:]
        dvel_dt = derivatives[:, half_dim:]
        velocities_new = velocities + dvel_dt * dt
        positions_new = positions + velocities_new * dt
        return torch.cat([positions_new, velocities_new], dim=-1)

    def integrate(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        if self.integration_method == 'rk4':
            return self.integrate_rk4(state, action)
        elif self.integration_method == 'dopri8':
            return self.integrate_dop853(state, action)
        elif self.integration_method == 'semi_implicit_euler':
            return self.integrate_semi_implicit_euler(state, action, dt=dt)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
