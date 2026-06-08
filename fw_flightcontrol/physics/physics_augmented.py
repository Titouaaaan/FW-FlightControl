import torch
import torch.nn as nn

class PhysicsAugmented(nn.Module):
    """Learned residual network F_a(s, u) that corrects physics prior predictions.

    Input:  [state, action, prev_action]  (all concatenated)
    Output: residual corrections to state time derivatives (same dim as state)

    Output layer is initialized with tiny weights (std=1e-4) so residuals start
    near zero, letting the physics prior dominate at the start of training.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 3, prev_action_dim: int = 0,
                 hidden_dims: list = None, activation: str = 'relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prev_action_dim = prev_action_dim
        self.output_dim = state_dim

        act_cls = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'elu': nn.ELU}.get(activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        input_dim = state_dim + action_dim + prev_action_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), act_cls()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        linear_layers = [l for l in self.network if isinstance(l, nn.Linear)]
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                if layer is linear_layers[-1]:
                    nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                prev_action: torch.Tensor = None) -> torch.Tensor:
        if prev_action is not None:
            x = torch.cat([state, action, prev_action], dim=-1)
        elif self.prev_action_dim > 0:
            pad = torch.zeros(*state.shape[:-1], self.prev_action_dim,
                              dtype=state.dtype, device=state.device)
            x = torch.cat([state, action, pad], dim=-1)
        else:
            x = torch.cat([state, action], dim=-1)
        return self.network(x)


class HybridDynamicsModel(nn.Module):
    """Combines physics prior and learned residual: ds/dt = F_p(s,u) + F_a(s,u).

    Supports ablation via with_prior / with_residual flags.
    Note: this forward operates in raw physical space without normalization.
    For training use HybridDynamicsODE (in training_objective.py) which handles
    normalization correctly.
    """

    def __init__(self, physics_prior, residual_network,
                 with_prior: bool = True, with_residual: bool = True):
        super().__init__()
        self.physics_prior = physics_prior
        self.residual_network = residual_network
        self.with_prior = with_prior
        self.with_residual = with_residual

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dx_dt = torch.zeros_like(state)
        if self.with_prior:
            dx_dt = self.physics_prior(state, action)
        if self.with_residual:
            residuals = self.residual_network(state, action)
            dx_dt = dx_dt + residuals if self.with_prior else residuals
        return dx_dt
