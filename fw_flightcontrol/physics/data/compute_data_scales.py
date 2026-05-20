#!/usr/bin/env python3
"""
Compute actual per-state scales from training data (mean absolute value).
These reflect what's actually in the data, not config bounds.
"""

import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(__file__).parent.parent / "data" / "updated_trajectory_data_progressive_noatmo.csv"
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

state_indices = [0, 1, 2, 3, 4, 5, 8, 9]
state_names = ['roll', 'pitch', 'airspeed_mps', 'p', 'q', 'r', 'alpha', 'beta']
state_cols = [f's_t_{i}' for i in state_indices]

print("\nPer-state statistics from training data:")
print("=" * 80)
print(f"{'State':<20} {'Mean |gt|':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Range/2':>12}")
print("-" * 80)

actual_scales = []
for name, col in zip(state_names, state_cols):
    values = df[col].values
    # Convert airspeed from km/h to m/s
    if 'airspeed' in name:
        values = values / 3.6
    
    mean_abs = np.mean(np.abs(values))
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    range_half = (max_val - min_val) / 2
    
    actual_scales.append(mean_abs)
    
    print(f"{name:<20} {mean_abs:>12.6f} {std:>12.6f} {min_val:>12.6f} {max_val:>12.6f} {range_half:>12.6f}")

print("=" * 80)
print("\nActual scales (mean |gt|) for use in per-state scaling:")
print(f"actual_scales = {actual_scales}")
print(f"as numpy array: np.array({actual_scales})")

print("\n" + "=" * 80)
print("Comparison: Config bounds vs. Actual data")
print("=" * 80)

config_bounds = {
    'roll': 3.1415927,
    'pitch': 3.1415927,
    'airspeed_mps': 55.55,
    'p': 35,
    'q': 35,
    'r': 35,
    'alpha': 0.35,
    'beta': 0.35,
}

print(f"{'State':<20} {'Config half-range':>20} {'Actual mean |gt|':>20} {'Ratio':>15}")
print("-" * 80)
for name, actual in zip(state_names, actual_scales):
    # Map state name to config key
    if name == 'airspeed_mps':
        config = config_bounds['airspeed_mps']
    else:
        config = config_bounds[name]
    ratio = config / actual if actual > 0 else 0
    print(f"{name:<20} {config:>20.6f} {actual:>20.6f} {ratio:>15.2f}x")

print("\n⚠️  Large ratios mean config bounds drastically over-estimate actual variation!")
print("This causes under-weighting when using config bounds for scaling.")
