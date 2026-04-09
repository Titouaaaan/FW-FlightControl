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
                 use_batch_norm: bool = False):
        """
        Initialize the residual network.
        
        Args:
            state_dim: Number of state dimensions (typically 8 for aircraft)
            action_dim: Number of action dimensions (3: aileron, elevator, throttle)
            hidden_dims: List of hidden layer sizes, e.g., [128, 64]
            activation: Activation function ('relu', 'tanh', 'elu')
            use_batch_norm: Whether to apply batch normalization between layers
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        
        # Build the MLP: [state, action] -> hidden layers -> residual corrections
        layers = []
        input_dim = state_dim + action_dim
        
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
        """
        Initialize network weights with small values.
        
        This is important for residual learning because we want the network
        to start near zero (i.e., no correction to physics prior initially),
        then gradually learn corrections during training.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute residual corrections to physics derivatives.
        
        Args:
            state: (batch_size, state_dim) - current state [φ, θ, Va, p, q, r, α, β]
            action: (batch_size, action_dim) - control input [δa, δe, throttle]
        
        Returns:
            residuals: (batch_size, state_dim) - correction to state time derivatives
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward through network
        residuals = self.network(x)
        
        return residuals


class HybridDynamicsModel(nn.Module):
    """
    Combines physics prior and learned residual network into a unified dynamics model.
    
    This allows for flexible ablation studies:
    - Physics only: with_prior=True, with_residual=False
    - Learning only: with_prior=False, with_residual=True
    - Hybrid: with_prior=True, with_residual=True (full model)
    
    The combined dynamics are: ds/dt = F_p(s,u) + F_a(s,u)
    """
    
    def __init__(self, 
                 physics_prior,
                 residual_network,
                 with_prior: bool = True,
                 with_residual: bool = True):
        """
        Initialize hybrid model.
        
        Args:
            physics_prior: PhysicsPrior instance (frozen, not trained)
            residual_network: PhysicsAugmented instance (trained)
            with_prior: Include physics prior in forward pass
            with_residual: Include learned residuals in forward pass
        """
        super().__init__()
        
        self.physics_prior = physics_prior
        self.residual_network = residual_network
        self.with_prior = with_prior
        self.with_residual = with_residual
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Combined forward pass: F = F_p + F_a (respecting ablation flags).
        
        Args:
            state: (batch_size, 8) state vector [φ, θ, Va, p, q, r, α, β]
            action: (batch_size, 3) action vector [δa, δe, throttle]
        
        Returns:
            dx_dt: (batch_size, 8) state time derivatives
        """
        dx_dt_combined = torch.zeros_like(state)
        
        # Add physics prior contribution
        if self.with_prior:
            dx_dt_physics = self.physics_prior(state, action)
            dx_dt_combined = dx_dt_physics
        
        # Add learned residual correction
        if self.with_residual:
            residuals = self.residual_network(state, action)
            if self.with_prior:
                # Sum: physics + residual
                dx_dt_combined = dx_dt_combined + residuals
            else:
                # Use residual only (ablation)
                dx_dt_combined = residuals
        
        return dx_dt_combined
    
    def integrate_rk4(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        RK4 integration for one environment step (0.01 seconds).
        
        We use torchdiffeq.odeint which provides automatic differentiation
        through the ODE integration, essential for computing gradients through
        multi-step predictions for training.
        
        Args:
            state: (batch_size, 8) initial state s_t
            action: (batch_size, 3) control input (constant during integration)
        
        Returns:
            state_next: (batch_size, 8) state after 0.01 seconds i.e., s_{t+1}
        """
        # Define the ODE: ds/dt = F_p(s,u) + F_a(s,u)
        def ode_dynamics(t, state_t):
            # t is unused (time-invariant system) but required by odeint API
            return self(state_t, action)
        
        # Integration time points: from t=0 to t=0.01
        t_eval = torch.tensor([0.0, 0.01], dtype=state.dtype, device=state.device)
        
        # Integrate using RK4 (4th order Runge-Kutta)
        # Returns shape (2, batch_size, 8) - one for each time point
        trajectory = odeint(ode_dynamics, state, t_eval, method='rk4')
        
        # Return final state at t=0.01
        return trajectory[-1]

