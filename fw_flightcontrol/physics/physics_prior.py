#!/usr/bin/env python3
import torch
import yaml
from pathlib import Path


class PhysicsPrior(torch.nn.Module):
    """Deterministic physics prior using aerodynamic equations."""

    def __init__(self, config_path: str = None):
        super().__init__()
        
        if config_path is None:
            config_path = str(Path(__file__).parent / 'aero_coefficients.yaml')
        
        self.load_config(config_path)
        
        for key, value in self.params.items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))

        self._precompute_gamma()

    def _precompute_gamma(self):
        Gamma = self.J_x * self.J_z - self.J_xz ** 2
        self.register_buffer('Gamma1', self.J_xz * (self.J_x - self.J_y + self.J_z) / Gamma)
        self.register_buffer('Gamma2', (self.J_z * (self.J_z - self.J_y) + self.J_xz ** 2) / Gamma)
        self.register_buffer('Gamma3', self.J_z / Gamma)
        self.register_buffer('Gamma4', self.J_xz / Gamma)
        self.register_buffer('Gamma5', (self.J_z - self.J_x) / self.J_y)
        self.register_buffer('Gamma6', self.J_xz / self.J_y)
        self.register_buffer('Gamma7', ((self.J_x - self.J_y) * self.J_x + self.J_xz ** 2) / Gamma)
        self.register_buffer('Gamma8', self.J_x / Gamma)
    
    def load_config(self, config_path: str):
        """Load aerodynamic coefficients and aircraft parameters from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.params = {}
        for key, value in config.items():
            if isinstance(value, (int, float)):
                self.params[key] = value
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivatives using physics prior.
        
        Args:
            state: (batch_size, 8) - [phi, theta, Va, p, q, r, alpha, beta]
            action: (batch_size, 3) - [delta_a, delta_e, throttle]
        
        Returns:
            dx_dt: (batch_size, 8) - time derivatives of state
        """
        batch_size = state.shape[0]
        
        # Unpack state
        phi = state[:, 0]        # roll angle (rad)
        theta = state[:, 1]      # pitch angle (rad)
        Va = state[:, 2]         # airspeed (m/s)
        p = state[:, 3]          # roll rate (rad/s)
        q = state[:, 4]          # pitch rate (rad/s)
        r = state[:, 5]          # yaw rate (rad/s)
        alpha = state[:, 6]      # angle of attack (rad)
        beta = state[:, 7]       # sideslip angle (rad)
        
        # Unpack action
        # JSBSim FCS scales the normalized command by ±30° before the aerodynamics:
        #   aerosurface_scale [-1,1] → [-30°,+30°] × 0.01745 rad/° = ±0.5235 rad
        # The aero coefficients (C_l_delta_a etc.) are per radian, so we must match that.
        delta_a  = action[:, 0] #* 0.5235  # [-1, 1] → [-0.5235, 0.5235] rad
        delta_e  = action[:, 1] #* 0.5235  # [-1, 1] → [-0.5235, 0.5235] rad
        throttle = action[:, 2]            # [0, 1]
        
        phi_dot = (p + 
                   torch.sin(phi) * torch.tan(theta) * q + 
                   torch.cos(phi) * torch.tan(theta) * r)
        
        theta_dot = torch.cos(phi) * q - torch.sin(phi) * r
        
        u = Va * torch.cos(alpha) * torch.cos(beta)
        v = Va * torch.sin(beta)
        w = Va * torch.sin(alpha) * torch.cos(beta)
        
        u_dot_kin = r * v - q * w
        v_dot_kin = p * w - r * u
        w_dot_kin = q * u - p * v
        
        f_x_g = -self.mass * self.g * torch.sin(theta)
        f_y_g = self.mass * self.g * torch.cos(theta) * torch.sin(phi)
        f_z_g = self.mass * self.g * torch.cos(theta) * torch.cos(phi)
        
        T_p = self.C_p * throttle  
        f_x_p = T_p
        f_y_p = torch.zeros_like(T_p)
        f_z_p = torch.zeros_like(T_p)
        
        q_dyn = 0.5 * self.rho * Va**2 * self.S  
        
        F_lift = q_dyn * (self.C_L0 + self.C_L_alpha * alpha + 
                         self.C_L_q * (self.c * q) / (2 * Va + 1e-6) +
                         self.C_L_delta_e * delta_e)
        
        F_drag = q_dyn * (self.C_D0 + self.C_D_alpha * alpha + 
                         self.C_D_delta_e * delta_e) 
        
        f_x_a = (-torch.cos(alpha) * F_drag + torch.sin(alpha) * F_lift)
        f_z_a = (-torch.sin(alpha) * F_drag - torch.cos(alpha) * F_lift)
        
        f_y_a = q_dyn * (self.C_Y0 + self.C_Y_beta * beta + 
                        self.C_Y_p * (self.b * p) / (2 * Va + 1e-6) +
                        self.C_Y_r * (self.b * r) / (2 * Va + 1e-6) +
                        self.C_Y_delta_a * delta_a)
        
        # Total forces
        f_x = f_x_g + f_x_a + f_x_p
        f_y = f_y_g + f_y_a + f_y_p
        f_z = f_z_g + f_z_a + f_z_p
        
        # Velocity time derivatives
        u_dot = u_dot_kin + f_x / self.mass
        v_dot = v_dot_kin + f_y / self.mass
        w_dot = w_dot_kin + f_z / self.mass
        
        Va_dot = (u * u_dot + v * v_dot + w * w_dot) / (Va + 1e-6)
        
        denom_alpha = u**2 + w**2 + 1e-6
        alpha_dot = (u * w_dot - w * u_dot) / denom_alpha
        
        denom_beta = Va * torch.sqrt(Va**2 - v**2 + 1e-6) + 1e-6
        beta_dot = (Va * v_dot - v * Va_dot) / denom_beta
        
        # Aerodynamic moments
        q_dyn_b = 0.5 * self.rho * Va**2 * self.S * self.b
        q_dyn_c = 0.5 * self.rho * Va**2 * self.S * self.c
        
        l = (self.C_l0 + self.C_l_beta * beta +
             self.C_l_p * (self.b * p) / (2 * Va + 1e-6) +
             self.C_l_r * (self.b * r) / (2 * Va + 1e-6) +
             self.C_l_delta_a * delta_a) * (q_dyn_b + 1e-6)
        
        m = (self.C_m0 + self.C_m_alpha * alpha +
             self.C_m_q * (self.c * q) / (2 * Va + 1e-6) +
             self.C_m_delta_e * delta_e) * (q_dyn_c + 1e-6)
        
        n = (self.C_n0 + self.C_n_beta * beta +
             self.C_n_p * (self.b * p) / (2 * Va + 1e-6) +
             self.C_n_r * (self.b * r) / (2 * Va + 1e-6) +
             self.C_n_delta_a * delta_a) * (q_dyn_b + 1e-6)
        
        p_dot = (self.Gamma1 * p * q - self.Gamma2 * q * r +
                 self.Gamma3 * l + self.Gamma4 * n)

        q_dot = self.Gamma5 * p * r - self.Gamma6 * (p**2 - r**2) + m / self.J_y

        r_dot = (self.Gamma7 * p * q - self.Gamma1 * q * r +
                 self.Gamma4 * l + self.Gamma8 * n)
        
        dx_dt = torch.stack([
            phi_dot, theta_dot, Va_dot, 
            p_dot, q_dot, r_dot, 
            alpha_dot, beta_dot
        ], dim=1)
        
        return dx_dt