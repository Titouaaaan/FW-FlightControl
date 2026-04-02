#!/usr/bin/env python3
"""
Test script for hybrid physics-augmented world model.

CORRECTED VERSION - Predicts s_t+1 exactly (0.01s, one environment step at 100 Hz)

This script:
1. Loads the physics prior F_p
2. Initializes the residual network F_a
3. Samples a transition from the trajectory dataset
4. Performs RK4 integration for exactly 0.01 seconds (one environment step)
5. Predicts all 8 state dimensions
6. Compares against ground truth s_t+1

No training yet - just validation that the forward pass works correctly.

Global Configuration:
  Set WITH_PRIOR and WITH_RESIDUAL to control which components are used during integration.
  - Physics prior only: WITH_PRIOR=True, WITH_RESIDUAL=False
  - Residual only: WITH_PRIOR=False, WITH_RESIDUAL=True  
  - Full hybrid: WITH_PRIOR=True, WITH_RESIDUAL=True
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

from physics_prior import PhysicsPrior
import physics_augmented
from physics_augmented import PhysicsAugmented, HybridDynamicsModel


# ============================================================================
# GLOBAL CONFIG - Control which components to include in integration
# ============================================================================
WITH_PRIOR = True        # Include physics prior F_p in integration
WITH_RESIDUAL = False     # Include residual network F_a in integration

# Synchronize with physics_augmented module
physics_augmented.WITH_PRIOR = WITH_PRIOR
physics_augmented.WITH_RESIDUAL = WITH_RESIDUAL
# ============================================================================

# State and action dimension names for clarity
STATE_DIMS = ['phi', 'theta', 'Va', 'p', 'q', 'r', 'alpha', 'beta']
ACTION_DIMS = ['aileron', 'elevator', 'throttle']


def load_sample_transition(csv_path: str = '../data/trajectory_data.csv', 
                           sample_idx: int = None):
    """
    Load a single transition from the CSV dataset.
    
    Args:
        csv_path: Path to trajectory CSV
        sample_idx: Index of transition (None = random sample)
    
    Returns:
        state_t: (8,) tensor - current state s_t
        action: (3,) tensor - action a_t
        state_t1: (8,) tensor - next state s_t+1 (ground truth)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(df))
    
    row = df.iloc[sample_idx]
    
    # Load all 8 state dimensions: [phi, theta, Va, p, q, r, alpha, beta]
    state_cols = [f's_t_{i}' for i in range(8)]
    state_t = torch.tensor([row[col] for col in state_cols], dtype=torch.float32)
    
    # Load all 3 action dimensions: [aileron, elevator, throttle]
    action_cols = [f'a_t_{i}' for i in range(3)]
    action = torch.tensor([row[col] for col in action_cols], dtype=torch.float32)
    
    # Load all 8 next state dimensions: [phi, theta, Va, p, q, r, alpha, beta]
    next_state_cols = [f's_t+1_{i}' for i in range(8)]
    state_t1 = torch.tensor([row[col] for col in next_state_cols], dtype=torch.float32)
    
    return state_t, action, state_t1


def test_forward_pass():
    """Test one forward pass through the hybrid model."""
    
    print("="*80)
    print("HYBRID PHYSICS-AUGMENTED MODEL TEST (CORRECTED)")
    print("="*80)
    print(f"\nObjective: Predict s_t+1 exactly (0.01s, one 100 Hz environment step)")
    print(f"Integration: 10 RK4 substeps × 0.001s = 0.01s total")
    print(f"Dimensions: Predict all 8 state values + all derivatives")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # ============ LOAD PHYSICS PRIOR ============
    print("\n1. Loading Physics Prior...")
    try:
        physics_prior = PhysicsPrior(config_path='aero_coefficients.yaml')
        physics_prior = physics_prior.to(device)
        physics_prior.eval()
        print("   ✓ Physics prior loaded successfully")
        print(f"   ✓ Aerodynamic coefficients loaded from YAML")
        
    except Exception as e:
        print(f"   ✗ Error loading physics prior: {e}")
        return
    
    # ============ INITIALIZE RESIDUAL NETWORK ============
    print("\n2. Initializing Residual Network...")
    residual_net = PhysicsAugmented(
        state_dim=8,
        action_dim=3,
        hidden_dims=[128, 128],
        activation='relu',
        use_batch_norm=False
    )
    residual_net = residual_net.to(device)
    residual_net.eval()
    n_params = sum(p.numel() for p in residual_net.parameters())
    print(f"   ✓ Residual network initialized")
    print(f"   ✓ Architecture: (8+3=11) -> 128 -> 128 -> 8")
    print(f"   ✓ Total parameters: {n_params:,}")
    
    # ============ CREATE HYBRID MODEL ============
    print("\n3. Creating Hybrid Model...")
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_net,
        dt=0.01  # Note: RK4 integration will use smaller substeps internally
    )
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()
    print("   ✓ Hybrid model (F_p + F_a) created")
    print(f"   ✓ Respects WITH_PRIOR={physics_augmented.WITH_PRIOR}, WITH_RESIDUAL={physics_augmented.WITH_RESIDUAL}")
    
    # ============ LOAD SAMPLE TRANSITION ============
    print("\n4. Loading Sample Transition from Dataset...")
    try:
        state_t, action, state_t1_true = load_sample_transition()
        state_t = state_t.unsqueeze(0).to(device)        # (1, 8)
        action = action.unsqueeze(0).to(device)          # (1, 3)
        state_t1_true = state_t1_true.to(device)         # (8,)
        
        print(f"   ✓ Sample loaded successfully")
        print(f"\n   Current State s_t (all 8 dimensions):")
        for i, name in enumerate(STATE_DIMS):
            print(f"     s_t[{i}] ({name:6s}): {state_t[0,i].item():+10.6f}")
        
        print(f"\n   Action a_t (all 3 dimensions):")
        for i, name in enumerate(ACTION_DIMS):
            print(f"     a_t[{i}] ({name:10s}): {action[0,i].item():+10.6f}")
            
        print(f"\n   Ground Truth: s_t+1 (all 8 dimensions):")
        for i, name in enumerate(STATE_DIMS):
            print(f"     s_t+1[{i}] ({name:6s}): {state_t1_true[i].item():+10.6f}")
            
    except Exception as e:
        print(f"   ✗ Error loading transition: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ EVALUATE PHYSICS PRIOR (F_p) ============
    print("\n5. Physics Prior Predictions (F_p only)...")
    with torch.no_grad():
        # Compute state derivatives F_p(s_t, a_t)
        dx_dt_physics = physics_prior(state_t, action)
        
        print(f"   ✓ Computed F_p(s_t, a_t)")
        print(f"     Shape: {dx_dt_physics.shape}")
        print(f"\n   State Time Derivatives (all 8 dimensions):")
        for i, name in enumerate(STATE_DIMS):
            print(f"     d({name:6s})/dt = {dx_dt_physics[0,i].item():+10.6f}")
        
        # RK4 integration to predict s_t+1 from s_t
        print(f"\n   RK4 Integration (predicting s_t+1 from s_t):")
        print(f"     Total integration time: 0.01 seconds")
        print(f"     RK4 substeps: 10 substeps × 0.001s each")
        
        state_pred_fp = state_t.clone()
        dt_substep = 0.001  # Each RK4 substep is 0.001 seconds
        num_substeps = 10   # Total: 10 × 0.001 = 0.01 seconds
        
        for step in range(num_substeps):
            # RK4 coefficients
            k1 = physics_prior(state_pred_fp, action)
            k2 = physics_prior(state_pred_fp + 0.5*dt_substep*k1, action)
            k3 = physics_prior(state_pred_fp + 0.5*dt_substep*k2, action)
            k4 = physics_prior(state_pred_fp + dt_substep*k3, action)
            
            # Check for NaN
            if torch.isnan(k1).any() or torch.isnan(k2).any() or torch.isnan(k3).any() or torch.isnan(k4).any():
                print(f"     ⚠️  Step {step}: NaN detected in derivatives!")
                state_pred_fp = torch.full_like(state_pred_fp, float('nan'))
                break
            
            # Update state
            state_increment = (dt_substep/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state_pred_fp = state_pred_fp + state_increment
            
            # Check for NaN in state
            if torch.isnan(state_pred_fp).any():
                print(f"     ⚠️  Step {step}: NaN in state after update!")
                break
        
        print(f"     ✓ Integration complete (10 substeps)")
        print(f"\n   Predicted Next State s_t+1 (using F_p only):")
        print(f"     All 8 dimensions:")
        errors_fp = []
        for i, name in enumerate(STATE_DIMS):
            pred = state_pred_fp[0,i].item()
            true = state_t1_true[i].item()
            error = pred - true
            errors_fp.append(error ** 2)
            print(f"       {name:6s}: pred={pred:+10.6f} | true={true:+10.6f} | error={error:+10.6f}")
        
        # Compute total error
        error_fp_total = torch.norm(state_pred_fp - state_t1_true.unsqueeze(0)).item()
        rel_error_fp = error_fp_total / (torch.norm(state_t1_true).item() + 1e-6)
        print(f"\n     L2 Error (all 8 dims): {error_fp_total:.6f}")
        print(f"     Relative Error: {rel_error_fp:.4%}")
    
    # ============ EVALUATE RESIDUAL NETWORK (F_a) ============
    print("\n6. Residual Network Predictions (F_a only, untrained)...")
    with torch.no_grad():
        # Compute residuals F_a(s_t, a_t)
        residuals_fa = residual_net(state_t, action)
        
        print(f"   ✓ Computed F_a(s_t, a_t)")
        print(f"     Shape: {residuals_fa.shape}")
        print(f"     L2 norm: {torch.norm(residuals_fa).item():.6f}")
        print(f"\n   Residual Corrections (all 8 dimensions):")
        for i, name in enumerate(STATE_DIMS):
            print(f"     F_a[{i}] ({name:6s}): {residuals_fa[0,i].item():+10.6f}")
    
    # ============ HYBRID MODEL INTEGRATION (F_p + F_a) ============
    print("\n7. Hybrid Model Integration (F_p + F_a)...")
    
    with torch.no_grad():
        print(f"     Configuration: WITH_PRIOR={physics_augmented.WITH_PRIOR}, WITH_RESIDUAL={physics_augmented.WITH_RESIDUAL}")
        
        # Integrate using hybrid model
        state_pred_hybrid = state_t.clone()
        dt_substep = 0.001
        num_substeps = 10
        
        for step in range(num_substeps):
            # RK4 with hybrid model (respects flags)
            k1 = hybrid_model(state_pred_hybrid, action)
            k2 = hybrid_model(state_pred_hybrid + 0.5*dt_substep*k1, action)
            k3 = hybrid_model(state_pred_hybrid + 0.5*dt_substep*k2, action)
            k4 = hybrid_model(state_pred_hybrid + dt_substep*k3, action)
            
            if torch.isnan(k1).any() or torch.isnan(k2).any() or torch.isnan(k3).any() or torch.isnan(k4).any():
                print(f"     ⚠️  Step {step}: NaN detected!")
                state_pred_hybrid = torch.full_like(state_pred_hybrid, float('nan'))
                break
            
            state_increment = (dt_substep/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state_pred_hybrid = state_pred_hybrid + state_increment
            
            if torch.isnan(state_pred_hybrid).any():
                print(f"     ⚠️  Step {step}: NaN in state after update!")
                break
        
        print(f"   ✓ Integration complete (10 substeps, 0.01s total)")
        print(f"\n   Predicted Next State s_t+1 (using F_p + F_a):")
        print(f"     All 8 dimensions:")
        errors_hybrid = []
        for i, name in enumerate(STATE_DIMS):
            pred = state_pred_hybrid[0,i].item()
            true = state_t1_true[i].item()
            error = pred - true
            errors_hybrid.append(error ** 2)
            print(f"       {name:6s}: pred={pred:+10.6f} | true={true:+10.6f} | error={error:+10.6f}")
        
        # Compute total error
        error_hybrid_total = torch.norm(state_pred_hybrid - state_t1_true.unsqueeze(0)).item()
        rel_error_hybrid = error_hybrid_total / (torch.norm(state_t1_true).item() + 1e-6)
        print(f"\n     L2 Error (all 8 dims): {error_hybrid_total:.6f}")
        print(f"     Relative Error: {rel_error_hybrid:.4%}")
    
    # ============ COMPARISON ACROSS ALL CONFIGURATIONS ============
    print("\n8. Ablation Study: Test All Flag Configurations...")
    print("\n   Configuration                       | L2 Error     | Rel Error")
    print("   " + "-"*65)
    
    with torch.no_grad():
        # Test all three configurations
        configs = [
            ("Physics Prior Only (F_p)", True, False),
            ("Residual Only (F_a)", False, True),
            ("Full Hybrid (F_p + F_a)", True, True),
        ]
        
        results = {}
        
        for config_name, with_prior, with_residual in configs:
            # Set flags
            physics_augmented.WITH_PRIOR = with_prior
            physics_augmented.WITH_RESIDUAL = with_residual
            
            # Integrate
            state_test = state_t.clone()
            dt_substep = 0.001
            num_substeps = 10
            
            try:
                for step in range(num_substeps):
                    k1 = hybrid_model(state_test, action)
                    k2 = hybrid_model(state_test + 0.5*dt_substep*k1, action)
                    k3 = hybrid_model(state_test + 0.5*dt_substep*k2, action)
                    k4 = hybrid_model(state_test + dt_substep*k3, action)
                    
                    if torch.isnan(k1).any() or torch.isnan(k2).any() or torch.isnan(k3).any() or torch.isnan(k4).any():
                        state_test = torch.full_like(state_test, float('nan'))
                        break
                    
                    state_increment = (dt_substep/6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    state_test = state_test + state_increment
                
                # Compute error
                if not torch.isnan(state_test).any() and not torch.isinf(state_test).any():
                    error = torch.norm(state_test - state_t1_true.unsqueeze(0)).item()
                    rel_error = error / (torch.norm(state_t1_true).item() + 1e-6)
                    results[config_name] = (error, rel_error, True)
                else:
                    results[config_name] = (float('inf'), float('inf'), False)
            
            except Exception as e:
                print(f"   Exception in {config_name}: {str(e)}")
                results[config_name] = (float('inf'), float('inf'), False)
        
        # Print results sorted by error
        for config_name, (error, rel_error, success) in sorted(results.items(), key=lambda x: x[1][0]):
            if success:
                print(f"   {config_name:35s} | {error:12.6f} | {rel_error:10.4%}")
            else:
                print(f"   {config_name:35s} | {'FAILED':12s} | {'':10s}")
        
        # Reset to original configuration
        physics_augmented.WITH_PRIOR = WITH_PRIOR
        physics_augmented.WITH_RESIDUAL = WITH_RESIDUAL
        
        # NOTE: Skipping "best configuration" display since network is untrained
        # The residual network (F_a) will appear best due to random weights,
        # but this is meaningless until the network is trained
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Successfully loaded s_t and predicted s_t+1")
    print(f"  - Integrated for exactly 0.01 seconds (one 100 Hz environment step)")
    print(f"  - Predicted all 8 state dimensions")
    print(f"  - Tested all flag configurations")
    print()


if __name__ == '__main__':
    test_forward_pass()
