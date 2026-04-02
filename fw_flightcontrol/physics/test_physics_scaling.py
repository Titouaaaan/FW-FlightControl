#!/usr/bin/env python3
"""
Test the physics prior with moment scaling enabled.
Verify that angular accelerations are now in a realistic range.
"""
import torch
import sys

# Ensure we're in the right directory
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics_prior import PhysicsPrior

# Initialize physics model
physics_prior = PhysicsPrior(
    config_path='/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/aero_coefficients.yaml'
)

# Test case 1: Max aileron deflection, moderate airspeed
state_1 = torch.tensor([
    [0.0, 0.05, 86.0, 0.0, 0.0, 0.0, 0.05, 0.0]  # [phi, theta, Va, p, q, r, alpha, beta]
], dtype=torch.float32)

action_1 = torch.tensor([
    [-1.0, 0.0, 0.5]  # [delta_a (max), delta_e (neutral), throttle]
], dtype=torch.float32)

# Test case 2: Max elevator deflection
state_2 = torch.tensor([
    [0.0, 0.05, 86.0, 0.0, 0.0, 0.0, 0.05, 0.0]
], dtype=torch.float32)

action_2 = torch.tensor([
    [0.0, 1.0, 0.5]  # [delta_a (neutral), delta_e (max), throttle]
], dtype=torch.float32)

# Compute derivatives
dx_dt_1 = physics_prior(state_1, action_1)
dx_dt_2 = physics_prior(state_2, action_2)

print("=" * 70)
print("PHYSICS PRIOR TEST WITH MOMENT SCALING")
print("=" * 70)
print()

print("TEST CASE 1: Max Aileron Deflection")
print("-" * 70)
print(f"State: phi={state_1[0,0]:.3f}, theta={state_1[0,1]:.3f}, Va={state_1[0,2]:.1f} m/s")
print(f"       p={state_1[0,3]:.3f}, q={state_1[0,4]:.3f}, r={state_1[0,5]:.3f} rad/s")
print(f"       alpha={state_1[0,6]:.3f}, beta={state_1[0,7]:.3f} rad")
print(f"Action: delta_a={action_1[0,0]:.2f}, delta_e={action_1[0,1]:.2f}, throttle={action_1[0,2]:.2f}")
print()
print(f"State derivatives (dx/dt):")
print(f"  phi_dot   = {dx_dt_1[0,0]:8.3f} rad/s")
print(f"  theta_dot = {dx_dt_1[0,1]:8.3f} rad/s")
print(f"  Va_dot    = {dx_dt_1[0,2]:8.3f} m/s²")
print(f"  p_dot     = {dx_dt_1[0,3]:8.3f} rad/s² <-- SHOULD BE ~0.01 to 5 rad/s²")
print(f"  q_dot     = {dx_dt_1[0,4]:8.3f} rad/s²")
print(f"  r_dot     = {dx_dt_1[0,5]:8.3f} rad/s²")
print(f"  alpha_dot = {dx_dt_1[0,6]:8.3f} rad/s")
print(f"  beta_dot  = {dx_dt_1[0,7]:8.3f} rad/s")
print()

print("TEST CASE 2: Max Elevator Deflection")
print("-" * 70)
print(f"State: phi={state_2[0,0]:.3f}, theta={state_2[0,1]:.3f}, Va={state_2[0,2]:.1f} m/s")
print(f"       p={state_2[0,3]:.3f}, q={state_2[0,4]:.3f}, r={state_2[0,5]:.3f} rad/s")
print(f"       alpha={state_2[0,6]:.3f}, beta={state_2[0,7]:.3f} rad")
print(f"Action: delta_a={action_2[0,0]:.2f}, delta_e={action_2[0,1]:.2f}, throttle={action_2[0,2]:.2f}")
print()
print(f"State derivatives (dx/dt):")
print(f"  phi_dot   = {dx_dt_2[0,0]:8.3f} rad/s")
print(f"  theta_dot = {dx_dt_2[0,1]:8.3f} rad/s")
print(f"  Va_dot    = {dx_dt_2[0,2]:8.3f} m/s²")
print(f"  p_dot     = {dx_dt_2[0,3]:8.3f} rad/s²")
print(f"  q_dot     = {dx_dt_2[0,4]:8.3f} rad/s² <-- SHOULD BE ~0.01 to 5 rad/s²")
print(f"  r_dot     = {dx_dt_2[0,5]:8.3f} rad/s²")
print(f"  alpha_dot = {dx_dt_2[0,6]:8.3f} rad/s")
print(f"  beta_dot  = {dx_dt_2[0,7]:8.3f} rad/s")
print()

# Assessment
print("ASSESSMENT:")
print("-" * 70)
p_dot_abs = abs(dx_dt_1[0,3].item())
q_dot_abs = abs(dx_dt_2[0,4].item())

if 0.01 <= p_dot_abs <= 10:
    print(f"✓ p_dot magnitude {p_dot_abs:.2f} rad/s² is in realistic range (0.01-10)")
else:
    print(f"✗ p_dot magnitude {p_dot_abs:.2f} rad/s² is outside realistic range (0.01-10)")
    if p_dot_abs > 10:
        print(f"  -> Consider reducing MOMENT_SCALING_FACTOR further")

if 0.01 <= q_dot_abs <= 10:
    print(f"✓ q_dot magnitude {q_dot_abs:.2f} rad/s² is in realistic range (0.01-10)")
else:
    print(f"✗ q_dot magnitude {q_dot_abs:.2f} rad/s² is outside realistic range (0.01-10)")
    if q_dot_abs > 10:
        print(f"  -> Consider reducing MOMENT_SCALING_FACTOR further")

print()
print("=" * 70)
