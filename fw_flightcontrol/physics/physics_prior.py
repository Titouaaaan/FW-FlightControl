#!/usr/bin/env python3
"""
Physics Prior F_p: Deterministic aerodynamic + gravity + propulsion dynamics.

Following the hybrid modeling framework from the thesis:
  x = [phi, theta, Va, p, q, r, alpha, beta] (8 dims)
  u = [delta_a, delta_e, throttle] (3 dims, aileron, elevator, and throttle)
  
Returns:
  dx/dt = F_p(x, u) (8 dims)

Key assumptions:
  - No wind (null or constant)
  - Fixed aerodynamic coefficients (from Skywalker X8 - Gryte et al. 2018)
  - Linear drag approximation
  - Throttle provided as 3rd action dimension (0 to 1)

CONFIGURATION FLAGS:
  WITH_PRIOR (line ~35): Include physics prior in hybrid model
  WITH_RESIDUAL (line ~36): Include learned residual in hybrid model
  APPLY_MOMENT_SCALING (line ~39): Apply empirical scaling to moment coefficients
  MOMENT_SCALING_FACTOR (line ~40): Scale factor for all moment coefficients (0.01 = 1% of original)
  
MOMENT SCALING NOTE:
  Gryte et al. 2018 aerodynamic coefficients are theoretically correct, but when combined
  with JSBSim inertia tensor values, produce unrealistically large angular accelerations
  (~600x too large). This is likely due to a mismatch between the coefficient definitions
  used in the paper vs. the inertia values from JSBSim. Empirical scaling compensates.
"""

import torch
import yaml
from pathlib import Path


# ===================== GLOBAL CONFIGURATION FLAGS =====================
# These flags enable/disable components for ablation studies and tuning

WITH_PRIOR = True           # Include physics prior model (F_p) in forward pass
WITH_RESIDUAL = False       # Include learned residual model (F_r) in forward pass

# Moment scaling (empirical calibration for linear model vs JSBSim nonlinear aerodynamics)
APPLY_MOMENT_SCALING = True  # Enable - necessary for linear approximation
MOMENT_SCALING_FACTOR = 1  # Scale all moments by this factor (0.002 = 1/500th original)
                               # Reason: Gryte et al. 2018 coefficients + JSBSim inertia produce
                               # ~600x too large angular accelerations. This scales to realistic values.
                               # TUNING: Adjust this value to get realistic physics:
                               #   - 0.001 = very docile aircraft (low authority, ~2-3 rad/s²)
                               #   - 0.002 = moderate control authority (~3-6 rad/s²)
                               #   - 0.003 = aggressive control (~6-10 rad/s²)
                               #   - Typical real aircraft: 2-5 rad/s² angular acceleration at max deflection


class PhysicsPrior(torch.nn.Module):
    """Deterministic physics prior using aerodynamic equations."""

    def __init__(self, config_path: str = 'aero_coefficients.yaml'):
        super().__init__()
        
        # Load aerodynamic coefficients and aircraft parameters
        self.load_config(config_path)
        
        # Register constants as buffers so they move with the model to GPU
        for key, value in self.params.items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))
    
    def load_config(self, config_path: str):
        """Load aerodynamic coefficients and aircraft parameters from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Separate into aerodynamic coefficients and physical parameters
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
        delta_a = action[:, 0]   # aileron deflection [-1, 1]
        delta_e = action[:, 1]   # elevator deflection [-1, 1]
        throttle = action[:, 2]  # throttle command [0, 1]
        
        # ===================== ATTITUDE KINEMATICS =====================
        # Eq: phi_dot = p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r
        phi_dot = (p + 
                   torch.sin(phi) * torch.tan(theta) * q + 
                   torch.cos(phi) * torch.tan(theta) * r)
        
        # Eq: theta_dot = cos(phi)*q - sin(phi)*r
        theta_dot = torch.cos(phi) * q - torch.sin(phi) * r
        
        # ===================== AIRSPEED DYNAMICS =====================
        # Body frame velocities from airspeed and angles
        u = Va * torch.cos(alpha) * torch.cos(beta)
        v = Va * torch.sin(beta)
        w = Va * torch.sin(alpha) * torch.cos(beta)
        
        # Euler angle rates for velocity components (kinematic coupling)
        u_dot_kin = r * v - q * w
        v_dot_kin = p * w - r * u
        w_dot_kin = q * u - p * v
        
        # ===================== FORCES (Gravity + Aero + Propulsion) =====================
        
        # Gravitational forces
        f_x_g = -self.mass * self.g * torch.sin(theta)
        f_y_g = self.mass * self.g * torch.cos(theta) * torch.sin(phi)
        f_z_g = self.mass * self.g * torch.cos(theta) * torch.cos(phi)
        
        # Propulsion force (thrust along x-axis)
        T_p = self.C_p * throttle  # Dynamic throttle from action
        f_x_p = T_p
        f_y_p = torch.zeros_like(T_p)
        f_z_p = torch.zeros_like(T_p)
        
        # ===================== AERODYNAMIC FORCES =====================
        q_dyn = 0.5 * self.rho * Va**2 * self.S  # Dynamic pressure * wing area
        
        # Lift and drag (using simple linear model)
        F_lift = q_dyn * (self.C_L0 + self.C_L_alpha * alpha + 
                         self.C_L_q * (self.c * q) / (2 * Va + 1e-6) +
                         self.C_L_delta_e * delta_e)
        
        F_drag = q_dyn * (self.C_D0 + self.C_D_alpha * alpha + 
                         self.C_D_delta_e * delta_e) # we remove the term that depends on C_D_q because it is set to zero
        
        # Transform lift/drag to body axes
        f_x_a = (-torch.cos(alpha) * F_drag + torch.sin(alpha) * F_lift)
        f_z_a = (-torch.sin(alpha) * F_drag - torch.cos(alpha) * F_lift)
        
        # Lateral aerodynamic force
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
        
        # ===================== AIRSPEED DERIVATIVE =====================
        # Va_dot = (u*u_dot + v*v_dot + w*w_dot) / Va
        Va_dot = (u * u_dot + v * v_dot + w * w_dot) / (Va + 1e-6)
        
        # ===================== ANGLE OF ATTACK DERIVATIVE =====================
        # alpha_dot = (u*w_dot - w*u_dot) / (u^2 + w^2)
        denom_alpha = u**2 + w**2 + 1e-6
        alpha_dot = (u * w_dot - w * u_dot) / denom_alpha
        
        # ===================== SIDESLIP ANGLE DERIVATIVE =====================
        # beta_dot = (Va*v_dot - v*Va_dot) / (Va * sqrt(Va^2 - v^2))
        denom_beta = Va * torch.sqrt(Va**2 - v**2 + 1e-6) + 1e-6
        beta_dot = (Va * v_dot - v * Va_dot) / denom_beta
        
        # ===================== ANGULAR RATE DERIVATIVES =====================
        
        # Aerodynamic moments
        q_dyn_b = 0.5 * self.rho * Va**2 * self.S * self.b
        q_dyn_c = 0.5 * self.rho * Va**2 * self.S * self.c
        
        l = q_dyn_b * (self.C_l0 + self.C_l_beta * beta +
                      self.C_l_p * (self.b * p) / (2 * Va + 1e-6) +
                      self.C_l_r * (self.b * r) / (2 * Va + 1e-6) +
                      self.C_l_delta_a * delta_a)
        
        m = q_dyn_c * (self.C_m0 + self.C_m_alpha * alpha +
                      self.C_m_q * (self.c * q) / (2 * Va + 1e-6) +
                      self.C_m_delta_e * delta_e)
        
        n = q_dyn_b * (self.C_n0 + self.C_n_beta * beta +
                      self.C_n_p * (self.b * p) / (2 * Va + 1e-6) +
                      self.C_n_r * (self.b * r) / (2 * Va + 1e-6) +
                      self.C_n_delta_a * delta_a)
        
        # Apply empirical moment scaling (calibration for coefficient definition mismatch)
        if APPLY_MOMENT_SCALING:
            l = l * MOMENT_SCALING_FACTOR
            m = m * MOMENT_SCALING_FACTOR
            n = n * MOMENT_SCALING_FACTOR
        
        # Compute Gamma parameters from inertia tensor
        # These come from inverting the inertia matrix for decoupled angular dynamics
        # Reference: Beard & McLain, Chapter 10 - equations for rigid body rotational dynamics
        J_x = self.J_x
        J_y = self.J_y
        J_z = self.J_z
        J_xz = self.J_xz
        
        # Primary denominator: Gamma = J_x * J_z - J_xz^2
        Gamma = J_x * J_z - J_xz**2
        
        # Gamma parameters for angular rate derivatives
        # Ref: User provided equations (from thesis)
        Gamma1 = J_xz * (J_x - J_y + J_z) / Gamma
        Gamma2 = (J_z * (J_z - J_y) + J_xz**2) / Gamma
        Gamma3 = J_z / Gamma
        Gamma4 = J_xz / Gamma
        Gamma5 = (J_z - J_x) / J_y
        Gamma6 = J_xz / J_y
        Gamma7 = ((J_x - J_y) * J_x + J_xz**2) / Gamma
        Gamma8 = J_x / Gamma
        
        # Angular rate derivatives
        p_dot = (Gamma1 * p * q - Gamma2 * q * r + 
                Gamma3 * l + Gamma4 * n)
        
        q_dot = Gamma5 * p * r - Gamma6 * (p**2 - r**2) + m / J_y
        
        r_dot = (Gamma7 * p * q - Gamma1 * q * r + 
                Gamma4 * l + Gamma8 * n)
        
        # ===================== STACK STATE DERIVATIVES =====================
        # Output: [phi_dot, theta_dot, Va_dot, p_dot, q_dot, r_dot, alpha_dot, beta_dot]
        dx_dt = torch.stack([
            phi_dot, theta_dot, Va_dot, 
            p_dot, q_dot, r_dot, 
            alpha_dot, beta_dot
        ], dim=1)
        
        return dx_dt
