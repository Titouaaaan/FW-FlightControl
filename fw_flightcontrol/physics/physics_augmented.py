#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchdiffeq import odeint


class PhysicsAugmented(nn.Module):
    """Learned residual network to augment physics prior."""
    
    def __init__(self, 
                 state_dim: int = 8,
                 action_dim: int = 2,
                 hidden_dims: list = None,
                 activation: str = 'relu',
                 use_batch_norm: bool = False):
        """
        Initialize the augmented physics network.
        
        Args:
            state_dim: Dimension of state vector (default 8)
            action_dim: Dimension of action vector (default 2)
            hidden_dims: List of hidden layer dimensions (default [128, 128])
            activation: Activation function name (relu, tanh, elu)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim  # Output same size as state derivatives
        
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
        
        # Build network: state + action -> hidden layers -> state residual
        layers = []
        input_dim = state_dim + action_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        # Output layer (no activation on output for residual learning)
        layers.append(nn.Linear(input_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small values for better early training."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Small initialization for residual learning
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute residual corrections to physics dynamics.
        
        Args:
            state: (batch_size, state_dim) - [phi, theta, Va, p, q, r, alpha, beta]
            action: (batch_size, action_dim) - [delta_a, delta_e, throttle] (typically 3D)
        
        Returns:
            residuals: (batch_size, state_dim) - corrections to state derivatives
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward through network
        residuals = self.network(x)
        
        return residuals


class HybridDynamicsModel(nn.Module):
    """Combined physics prior + learned augmentation with instance-level flags for ablation studies."""
    
    def __init__(self, 
                 physics_prior,
                 residual_network,
                 with_prior: bool = True,
                 with_residual: bool = True):
        """
        Initialize hybrid dynamics model.
        
        Args:
            physics_prior: PhysicsPrior instance that returns state derivatives F_p(s, u)
            residual_network: PhysicsAugmented instance that returns residual corrections F_a(s, u) of derivatives
            with_prior: Include physics prior in forward pass
            with_residual: Include learned residual in forward pass
        """
        super().__init__()
        
        self.physics_prior = physics_prior
        self.residual_network = residual_network
        self.with_prior = with_prior
        self.with_residual = with_residual
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Combined dynamics: F = F_p + F_a (respecting instance flags).
        
        Supports ablation studies:
        - Physics prior only: with_prior=True, with_residual=False
        - Residual only: with_prior=False, with_residual=True
        - Full hybrid: with_prior=True, with_residual=True
        
        Args:
            state: (batch_size, 8) state vector [phi, theta, Va, p, q, r, alpha, beta]
            action: (batch_size, 2 or 3) action vector
        
        Returns:
            dx_dt: (batch_size, 8) state time derivatives
        """
        dx_dt_combined = torch.zeros_like(state)
        
        # Physics prior prediction (if enabled)
        if self.with_prior:
            dx_dt_physics = self.physics_prior(state, action)
            dx_dt_combined = dx_dt_physics
        
        # Learned residual correction (if enabled)
        if self.with_residual:
            residuals = self.residual_network(state, action)
            if self.with_prior:
                # Add residuals to physics prior
                dx_dt_combined = dx_dt_combined + residuals
            else:
                # Use only residuals
                dx_dt_combined = residuals
        
        return dx_dt_combined
    
    def integrate_rk4(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        RK4 integration using combined dynamics for one environment step.
        
        Integrates from s_t to s_t+1 (one 100 Hz environment step = 0.01 seconds).
        Uses torchdiffeq.odeint with RK4 method for stable integration with automatic
        differentiation support (crucial for training residual network).
        Respects instance flags with_prior and with_residual.
        
        Args:
            state: (batch_size, 8) initial state s_t
            action: (batch_size, 3) action vector [delta_a, delta_e, throttle] (constant over integration)
        
        Returns:
            state_next: (batch_size, 8) state after 0.01 seconds integration (s_t+1)
        """
        # Define ODE dynamics function for torchdiffeq
        def ode_dynamics(t, state_t):
            # t is unused (time-invariant system), but required by odeint API
            return self(state_t, action)
        
        # Integration times: start at t=0, end at t=0.01
        t_eval = torch.tensor([0.0, 0.01], dtype=state.dtype, device=state.device)
        
        # Integrate using RK4 method
        # Returns trajectory shape (2, batch_size, 8) - one for each time point
        trajectory = odeint(ode_dynamics, state, t_eval, method='rk4')
        
        # Return final state at t=0.01
        return trajectory[-1]
