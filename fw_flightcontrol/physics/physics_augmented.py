#!/usr/bin/env python3
"""
Physics-Augmented Network F_a: Learned residual corrections to physics prior.

The combined model is F = F_p + F_a, where:
  F_p: deterministic physics (equations of motion)
  F_a: learned neural network (captures unmodeled phenomena)

This follows the APHYNITY framework for error compounding during training.

State variables: [phi, theta, Va, p, q, r, alpha, beta] (8 dims)
Action variables: [delta_a, delta_e] (2 dims) or [delta_a, delta_e, throttle] (3 dims)
Output: Residual corrections to state derivatives (8 dims)
"""

import torch
import torch.nn as nn

# Global configuration flags (set by learn_physics_model.py or training scripts)
# These allow selective testing of physics prior vs. residual network components
WITH_PRIOR = True        # Include physics prior F_p in computations
WITH_RESIDUAL = True     # Include residual network F_a in computations


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
    """Combined physics prior + learned augmentation."""
    
    def __init__(self, 
                 physics_prior,
                 residual_network,
                 dt: float = 0.01):
        """
        Initialize hybrid dynamics model.
        
        Args:
            physics_prior: PhysicsPrior instance (F_p)
            residual_network: PhysicsAugmented instance (F_a)
            dt: Integration timestep for RK4
        """
        super().__init__()
        
        self.physics_prior = physics_prior
        self.residual_network = residual_network
        self.dt = dt
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Combined dynamics: F = F_p + F_a with flag control.
        
        Respects global flags WITH_PRIOR and WITH_RESIDUAL to allow:
        - Physics prior only: WITH_PRIOR=True, WITH_RESIDUAL=False
        - Residual only: WITH_PRIOR=False, WITH_RESIDUAL=True
        - Full hybrid: WITH_PRIOR=True, WITH_RESIDUAL=True
        
        Args:
            state: (batch_size, 8) state vector
            action: (batch_size, 2 or 3) action vector
        
        Returns:
            dx_dt: (batch_size, 8) state time derivatives
        """
        dx_dt_combined = torch.zeros_like(state)
        
        # Physics prior prediction (if enabled)
        if WITH_PRIOR:
            dx_dt_physics = self.physics_prior(state, action)
            dx_dt_combined = dx_dt_physics
        
        # Learned residual correction (if enabled)
        if WITH_RESIDUAL:
            residuals = self.residual_network(state, action)
            if WITH_PRIOR:
                # Add residuals to physics prior
                dx_dt_combined = dx_dt_combined + residuals
            else:
                # Use only residuals
                dx_dt_combined = residuals
        
        return dx_dt_combined
    
    def integrate_rk4(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        RK4 integration using combined dynamics (respects global flags).
        
        Performs 100 substeps of RK4 integration (dt=0.01 each) for a total of
        1.0 second integration. The combined dynamics uses:
        - Physics prior only if WITH_PRIOR=True, WITH_RESIDUAL=False
        - Residual only if WITH_PRIOR=False, WITH_RESIDUAL=True
        - Full hybrid (F_p + F_a) if both flags are True
        
        Args:
            state: (batch_size, 8) initial state
            action: (batch_size, 3) action vector [delta_a, delta_e, throttle] (constant over timestep)
        
        Returns:
            state_next: (batch_size, 8) state after 100 RK4 substeps (1.0 second total)
        """
        dt = self.dt
        state_integrated = state.clone()
        
        # 100 substeps of RK4 integration
        for _ in range(100):
            # K1: derivatives at current state
            k1 = self(state_integrated, action)
            
            # K2: derivatives at midpoint using K1
            k2 = self(state_integrated + 0.5 * dt * k1, action)
            
            # K3: derivatives at midpoint using K2
            k3 = self(state_integrated + 0.5 * dt * k2, action)
            
            # K4: derivatives at next point using K3
            k4 = self(state_integrated + dt * k3, action)
            
            # RK4 integration step
            state_integrated = state_integrated + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return state_integrated
