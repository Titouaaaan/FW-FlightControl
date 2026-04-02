#!/usr/bin/env python3
"""
Hypothesis test: Check if moment coefficients have S×b factor included
"""
import torch
import yaml

# Load parameters
with open('aero_coefficients.yaml', 'r') as f:
    params = yaml.safe_load(f)

Va = 86.0  # m/s (from test sample)
delta_a = -0.27  # radians (typical aileron deflection)
rho = params['rho']
S = params['S']
b = params['b']
c = params['c']

# Current calculation (with S×b included)
q_dyn = 0.5 * rho * Va**2
q_dyn_b = 0.5 * rho * Va**2 * S * b
q_dyn_c = 0.5 * rho * Va**2 * S * c

l_current = q_dyn_b * params['C_l_delta_a'] * delta_a
m_current = q_dyn_c * params['C_m_delta_e'] * 0.2  # assume delta_e = 0.2 rad
n_current = q_dyn_b * params['C_n_delta_a'] * delta_a

print("=" * 60)
print("CURRENT CALCULATION (with S×b included)")
print("=" * 60)
print(f"q_dyn = 0.5 × ρ × Va² = {q_dyn:.1f} Pa")
print(f"q_dyn_b = {q_dyn_b:.1f} N")
print(f"q_dyn_c = {q_dyn_c:.1f} N")
print()
print(f"Rolling moment:  l = {l_current:.1f} N·m")
print(f"Pitching moment: m = {m_current:.1f} N·m")
print(f"Yawing moment:   n = {n_current:.1f} N·m")
print()

# Hypothesis: divide by S×b (coefficients don't include these factors)
l_hypothesis = q_dyn * params['C_l_delta_a'] * delta_a
m_hypothesis = q_dyn * params['C_m_delta_e'] * 0.2
n_hypothesis = q_dyn * params['C_n_delta_a'] * delta_a

print("=" * 60)
print("HYPOTHESIS: Coefficients DON'T include S×b")
print("(Divide all moments by S×b = {:.3f})".format(S * b))
print("=" * 60)
print(f"Rolling moment:  l = {l_hypothesis:.1f} N·m (ratio: {l_hypothesis/l_current:.3f})")
print(f"Pitching moment: m = {m_hypothesis:.1f} N·m (ratio: {m_hypothesis/m_current:.3f})")
print(f"Yawing moment:   n = {n_hypothesis:.1f} N·m (ratio: {n_hypothesis/n_current:.3f})")
print()

# Angular accelerations with current inertia
J_x = params['J_x']
J_y = params['J_y']
J_z = params['J_z']
J_xz = params['J_xz']

Gamma = J_x * J_z - J_xz**2
Gamma3 = J_z / Gamma

p_dot_current = Gamma3 * l_current
q_dot_current = m_current / J_y  # simplified
r_dot_current = Gamma3 * n_current

p_dot_hypothesis = Gamma3 * l_hypothesis
q_dot_hypothesis = m_hypothesis / J_y
r_dot_hypothesis = Gamma3 * n_hypothesis

print("=" * 60)
print("ANGULAR ACCELERATIONS")
print("=" * 60)
print(f"Current approach:")
print(f"  p_dot = {p_dot_current:8.1f} rad/s² (UNREALISTIC)")
print(f"  q_dot = {q_dot_current:8.1f} rad/s²")
print(f"  r_dot = {r_dot_current:8.1f} rad/s² (UNREALISTIC)")
print()
print(f"If hypothesis correct:")
print(f"  p_dot = {p_dot_hypothesis:8.1f} rad/s² (more realistic?)")
print(f"  q_dot = {q_dot_hypothesis:8.1f} rad/s²")
print(f"  r_dot = {r_dot_hypothesis:8.1f} rad/s² (more realistic?)")
print()
print(f"Recommended range: -10 to +10 rad/s² for max control input")
print()

# Test another hypothesis: maybe only body-fitted coefficients need the scaling
# E.g., C_l, C_m, C_n are per unit dynamic pressure, not per dynamic pressure × area
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Ratio of moments (hypothesis / current): {l_hypothesis/l_current:.3f}")
print(f"Ratio of angular accel (hypothesis / current): {p_dot_hypothesis/p_dot_current:.3f}")
print()
print("If hypothesis gives p_dot ~ 120-200, then moment equation")
print("in Gryte paper likely uses q_dyn directly, not q_dyn×S×b")
