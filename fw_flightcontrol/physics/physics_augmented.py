import torch
import torch.nn as nn
from torchdiffeq import odeint


class PhysicsAugmented(nn.Module):

    def __init__(self,
                 state_dim: int = 8,
                 action_dim: int = 3,
                 hidden_dims: list = None,
                 activation: str = 'relu',
                 prev_action_dim: int = 0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prev_action_dim = prev_action_dim
        self.output_dim = state_dim

        act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'elu': nn.ELU}
        act_cls = act_map.get(activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        input_dim = state_dim + action_dim + prev_action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_cls())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                is_output = (i == len(self.network) - 1)
                if is_output:
                    nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                prev_action: torch.Tensor = None) -> torch.Tensor:
        parts = [state, action]
        if prev_action is not None:
            parts.append(prev_action)
        x = torch.cat(parts, dim=-1)
        return self.network(x)


class HybridDynamicsModel(nn.Module):

    def __init__(self,
                 physics_prior,
                 residual_network,
                 with_prior: bool = True,
                 with_residual: bool = True):
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
