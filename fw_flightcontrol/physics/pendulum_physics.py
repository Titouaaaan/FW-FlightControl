import torch
import torch.nn as nn
import math


class PendulumPhysics(nn.Module):
    
    def __init__(self, omega0_square=None, alpha=0.2):
        super().__init__()
        if omega0_square is None:
            omega0_square = (2 * math.pi / 6) ** 2
        self.omega0_square = omega0_square
        self.alpha = alpha
        self.register_buffer('_omega0_sq', torch.tensor(omega0_square, dtype=torch.float32))
        self.register_buffer('_alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, state, action):
        theta = state[:, 0]
        omega = state[:, 1]
        dtheta_dt = omega
        domega_dt = -(self._omega0_sq * torch.sin(theta)) - (self._alpha * omega)
        derivative = torch.stack([dtheta_dt, domega_dt], dim=1)
        return derivative

    def to(self, device):
        super().to(device)
        return self
