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


class PhysicsAugmented(nn.Module):
    """
    Learned residual network that corrects physics prior predictions.

    Input: current state s (normalized) and action u
    Output: residual corrections Δ(ds/dt) in normalized space

    We use a simple MLP architecture for efficiency and stability.
    The network is initialized with small weights to ensure the residuals
    start near zero, letting the physics prior dominate early training.
    """

    def __init__(self,
                 state_dim: int = 8,
                 action_dim: int = 3,
                 hidden_dims: list = None,
                 activation: str = 'relu',
                 use_batch_norm: bool = False):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.use_batch_norm = use_batch_norm

        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))


class HybridDynamicsModel(nn.Module):
    """
    Combines physics prior and learned residual network into a unified dynamics model.

    Ablation flags:
    - with_prior=True,  with_residual=False : physics only
    - with_prior=False, with_residual=True  : residual only
    - with_prior=True,  with_residual=True  : full hybrid model (default)

    The combined dynamics are: ds_raw/dt = F_p(s_raw, u) + F_a(s_norm, u) * std

    norm_scale and norm_offset are set after checkpoint loading and travel with
    the model. Integration is handled externally via HybridDynamicsODE.
    """

    def __init__(self,
                 physics_prior,
                 residual_network,
                 with_prior: bool = True,
                 with_residual: bool = True,
                 integration_method: str = 'rk4'):
        super().__init__()

        self.physics_prior     = physics_prior
        self.residual_network  = residual_network
        self.with_prior        = with_prior
        self.with_residual     = with_residual
        self.integration_method = integration_method

        if integration_method not in ['rk4', 'dopri8', 'semi_implicit_euler']:
            raise ValueError(
                f"integration_method must be 'rk4', 'dopri8', or 'semi_implicit_euler', "
                f"got '{integration_method}'"
            )

        # Normalization parameters — set externally after checkpoint loading.
        # Plain attributes (not buffers) so they don't interfere with state_dict.
        # Must be on the same device as the model when used.
        self.norm_scale  = None  # std  per state dimension
        self.norm_offset = None  # mean per state dimension

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute combined state derivative in raw physical space.

        Expects state in raw (physical) units. The residual network receives the
        normalized state internally; its output is scaled back to raw units.
        """
        dx_dt = torch.zeros_like(state)

        if self.with_prior:
            dx_dt = self.physics_prior(state, action)

        if self.with_residual:
            state_norm    = (state - self.norm_offset) / self.norm_scale
            residual_out  = self.residual_network(state_norm, action)
            dx_dt         = dx_dt + residual_out * self.norm_scale

        return dx_dt
