#!/usr/bin/env python3
"""
Multi-sample physics prior investigation.

Tests the physics prior on multiple random samples from the dataset
to identify systematic errors in the aerodynamic model.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

from physics_prior import PhysicsPrior
from physics_augmented import PhysicsAugmented, HybridDynamicsModel


STATE_DIMS = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
ACTION_DIMS = ['aileron', 'elevator', 'throttle']


def load_sample_transition(csv_path: str = '../data/trajectory_data.csv', sample_idx: int = None):
    """Load a single transition from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(df))
    
    row = df.iloc[sample_idx]
    
    state_cols = [f's_t_{i}' for i in range(8)]
    state_t = torch.tensor([row[col] for col in state_cols], dtype=torch.float32)
    
    action_cols = [f'a_t_{i}' for i in range(3)]
    action = torch.tensor([row[col] for col in action_cols], dtype=torch.float32)
    
    next_state_cols = [f's_t+1_{i}' for i in range(8)]
    state_t1 = torch.tensor([row[col] for col in next_state_cols], dtype=torch.float32)
    
    return state_t, action, state_t1


def test_physics_prior_on_sample(physics_prior, hybrid_model, state_t, action, state_t1_true, device):
    """Test physics prior on a single sample and return detailed error analysis."""
    
    with torch.no_grad():
        state_t = state_t.unsqueeze(0).to(device)
        action = action.unsqueeze(0).to(device)
        state_t1_true = state_t1_true.to(device)
        
        # Compute derivatives
        dx_dt = physics_prior(state_t, action)
        
        # RK4 integration
        state_pred = state_t.clone()
        dt_substep = 0.001
        num_substeps = 10
        
        for step in range(num_substeps):
            k1 = physics_prior(state_pred, action)
            k2 = physics_prior(state_pred + 0.5*dt_substep*k1, action)
            k3 = physics_prior(state_pred + 0.5*dt_substep*k2, action)
            k4 = physics_prior(state_pred + dt_substep*k3, action)
            
            if torch.isnan(k1).any():
                return None
            
            state_increment = (dt_substep/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state_pred = state_pred + state_increment
        
        if torch.isnan(state_pred).any():
            return None
        
        # Compute errors per dimension
        errors = {}
        for i, name in enumerate(STATE_DIMS):
            pred = state_pred[0, i].item()
            true = state_t1_true[i].item()
            delta = true - state_t[0, i].item()  # Ground truth change
            error = pred - true
            rel_error = error / (abs(delta) + 1e-6) if abs(delta) > 1e-3 else error
            
            errors[name] = {
                'pred': pred,
                'true': true,
                'error': error,
                'rel_error': rel_error,
                'delta': delta,
                'deriv': dx_dt[0, i].item()
            }
        
        total_error = torch.norm(state_pred - state_t1_true.unsqueeze(0)).item()
        
        return errors, total_error


def main():
    """Run investigation on multiple samples."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*90)
    print("PHYSICS PRIOR INVESTIGATION - Multi-Sample Analysis")
    print("="*90)
    
    # Load models
    print("\nLoading models...")
    physics_prior = PhysicsPrior(config_path='aero_coefficients.yaml')
    physics_prior = physics_prior.to(device).eval()
    
    residual_net = PhysicsAugmented(state_dim=8, action_dim=3, hidden_dims=[128, 128])
    residual_net = residual_net.to(device).eval()
    
    hybrid_model = HybridDynamicsModel(physics_prior, residual_net, dt=0.01)
    hybrid_model = hybrid_model.to(device).eval()
    
    print("✓ Models loaded\n")
    
    # Test on multiple random samples
    num_samples = 5
    all_results = []
    
    for sample_num in range(num_samples):
        sample_idx = np.random.randint(0, 64000)
        state_t, action, state_t1 = load_sample_transition(sample_idx=sample_idx)
        
        result = test_physics_prior_on_sample(physics_prior, hybrid_model, state_t, action, state_t1, device)
        
        if result is None:
            print(f"Sample {sample_num+1} (idx {sample_idx}): FAILED (NaN)")
            continue
        
        errors, total_error = result
        all_results.append((sample_idx, errors, total_error))
        
        print(f"\n{'='*90}")
        print(f"Sample {sample_num+1} (Dataset Index {sample_idx})")
        print(f"{'='*90}")
        
        print(f"\nInitial State s_t:")
        for i, name in enumerate(STATE_DIMS):
            print(f"  {name:6s}: {state_t[i].item():+9.4f}")
        
        print(f"\nAction a_t:")
        for i, name in enumerate(ACTION_DIMS):
            print(f"  {name:10s}: {action[i].item():+9.4f}")
        
        print(f"\nDerivatives F_p(s_t, a_t):")
        for name in STATE_DIMS:
            deriv = errors[name]['deriv']
            print(f"  d({name:6s})/dt: {deriv:+10.4f}")
        
        print(f"\nState Prediction Errors (0.01s integration):")
        print(f"  {'Dimension':<10} {'Predicted':>12} {'Ground Truth':>12} {'Error':>12} {'Delta':>12} {'Rel Error':>12}")
        print(f"  {'-'*90}")
        
        for name in STATE_DIMS:
            e = errors[name]
            print(f"  {name:<10} {e['pred']:>12.6f} {e['true']:>12.6f} {e['error']:>12.6f} {e['delta']:>12.6f} {e['rel_error']:>12.4f}")
        
        print(f"\nTotal L2 Error: {total_error:.6f}")
    
    # Analysis across all samples
    print(f"\n\n{'='*90}")
    print("CROSS-SAMPLE ANALYSIS")
    print(f"{'='*90}")
    
    # Aggregate error statistics by dimension
    dim_errors = {name: [] for name in STATE_DIMS}
    dim_rel_errors = {name: [] for name in STATE_DIMS}
    
    for sample_idx, errors, total_error in all_results:
        for name in STATE_DIMS:
            dim_errors[name].append(errors[name]['error'])
            dim_rel_errors[name].append(abs(errors[name]['rel_error']))
    
    print(f"\nError Statistics by Dimension:")
    print(f"  {'Dimension':<10} {'Mean Error':>15} {'Std Error':>15} {'Mean |Rel Error|':>15}")
    print(f"  {'-'*60}")
    
    for name in STATE_DIMS:
        mean_err = np.mean(dim_errors[name])
        std_err = np.std(dim_errors[name])
        mean_rel_err = np.mean(dim_rel_errors[name])
        print(f"  {name:<10} {mean_err:>15.6f} {std_err:>15.6f} {mean_rel_err:>15.6f}")
    
    # Identify problematic dimensions
    print(f"\nProblematic Dimensions (mean |error| > 0.1):")
    for name in STATE_DIMS:
        mean_err = np.mean(np.abs(dim_errors[name]))
        if mean_err > 0.1:
            print(f"  ⚠️  {name}: {mean_err:.6f}")
    
    print("\n" + "="*90)


if __name__ == '__main__':
    main()
