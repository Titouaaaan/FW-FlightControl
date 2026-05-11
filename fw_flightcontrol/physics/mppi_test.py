#!/usr/bin/env python3
"""
MPPI (Model Predictive Path Integral) Control for Attitude Tracking

This script demonstrates online trajectory optimization using the hybrid physics-augmented
dynamics model to predict future states, combined with MPPI for action selection.

Since no reward model is learned, we use roll and pitch tracking errors as the cost function.
The MPPI algorithm samples multiple action trajectories, evaluates them using the model,
and returns the action with the lowest tracking cost.

Usage:
    python mppi_test.py --checkpoint path/to/model.pt --config path/to/training_params.yaml
    python mppi_test.py --checkpoint final_model.pt
"""

import torch
import numpy as np
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import gymnasium as gym
import fw_jsbgym  # registers JSBSim gym environments
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project to path
sys.path.insert(0, '/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl')

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import HybridDynamicsODE
from fw_flightcontrol.agents.pid import PID
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_flightcontrol.physics.utils import (
    load_config, get_norm_type, normalize_state_torch, denormalize_state_torch,
    extract_bounds_from_config, compute_denorm_factors, compute_data_norm_params,
    clean_state_dict_for_compilation, compute_convergence_stats,
    plot_tracking_performance, plot_action_history,
)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_config_and_model(
    model_path: str,
    config_path: str,
    device: torch.device
) -> Tuple[HybridDynamicsModel, Dict, torch.Tensor, torch.Tensor, Optional[str]]:
    """
    Load training configuration and initialize the hybrid model.
    
    Args:
        model_path: Path to trained residual network checkpoint
        config_path: Path to training_params.yaml config file
        device: Device to load model on (cuda or cpu)
    
    Returns:
        Tuple of:
        - hybrid_model: HybridDynamicsModel instance (eval mode)
        - config: Configuration dictionary
        - denorm_factors: Torch tensor for denormalization scaling
        - min_bounds: Torch tensor for denormalization offset
        - norm_type: Normalization type string or None
    """
    print("\n" + "="*60)
    print("LOADING MODEL AND CONFIGURATION")
    print("="*60)
    
    # Load configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Initialize physics prior (frozen, not trained)
    physics_prior = PhysicsPrior()
    print("  ✓ Physics prior initialized (frozen)")
    
    # Load checkpoint first so we can infer the architecture from the actual weights
    print(f"Loading checkpoint from: {model_path}")
    raw = torch.load(model_path, map_location=device)

    denorm_factors = None
    min_bounds = None

    if isinstance(raw, dict) and 'residual_state' in raw:
        residual_state = clean_state_dict_for_compilation(raw['residual_state'])
        saved_epoch = raw.get('epoch', '?')
        saved_lambda = raw.get('lambda', '?')
        if 'norm_scale' in raw and 'norm_offset' in raw:
            denorm_factors = torch.tensor(raw['norm_scale'], dtype=torch.float32, device=device)
            min_bounds = torch.tensor(raw['norm_offset'], dtype=torch.float32, device=device)
            print(f"  ✓ Loaded normalization parameters from checkpoint")
    else:
        residual_state = clean_state_dict_for_compilation(raw)

    # Infer hidden_dims from the saved weights so the model always matches the checkpoint,
    # even when training_params.yaml has been updated to a different architecture.
    def _infer_hidden_dims(sd):
        dims = []
        i = 0
        while f'network.{i}.weight' in sd:
            if f'network.{i+2}.weight' in sd:
                dims.append(sd[f'network.{i}.weight'].shape[0])
            i += 2
        return dims

    net_config = config['network']
    inferred_hidden = _infer_hidden_dims(residual_state)
    if inferred_hidden != net_config['hidden_dims']:
        print(f"  ⚠ Architecture mismatch: config says {net_config['hidden_dims']}, "
              f"checkpoint has {inferred_hidden}. Using checkpoint architecture.")

    # Initialize residual network with architecture that matches the checkpoint
    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=inferred_hidden if inferred_hidden else net_config['hidden_dims'],
        activation=net_config['activation'],
        use_batch_norm=net_config['use_batch_norm'],
    )
    residual_network.load_state_dict(residual_state)

    if isinstance(raw, dict) and 'residual_state' in raw:
        print(f"  ✓ Loaded from training checkpoint (epoch={saved_epoch}, λ={saved_lambda})")
    else:
        print("  ✓ Loaded bare state dict")
    
    num_params = sum(p.numel() for p in residual_network.parameters())
    print(f"  ✓ Residual network: {num_params:,} parameters")
    
    # Create hybrid model
    integration_method = config.get('integration', {}).get('method', 'rk4')
    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True,
        integration_method=integration_method,
    )
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()
    print(f"  ✓ Hybrid model ready on {device}")
    
    # Extract normalization type
    norm_type = get_norm_type(config)
    print(f"  Normalization type: {norm_type if norm_type else 'None (raw space)'}")
    
    # If norm parameters not in checkpoint, try to extract from config
    if norm_type is not None and denorm_factors is None:
        if 'normalization_params' in config:
            p = config['normalization_params']
            denorm_factors = torch.tensor(p['norm_scale'], dtype=torch.float32, device=device)
            min_bounds = torch.tensor(p['norm_offset'], dtype=torch.float32, device=device)
            print(f"  ✓ Loaded normalization parameters from config")
        else:
            print(f"  ⚠ WARNING: No normalization parameters found in checkpoint or config!")
            print(f"  ⚠ Cannot apply {norm_type} normalization without these parameters.")
            norm_type = None
    
    print("="*60 + "\n")
    return hybrid_model, config, denorm_factors, min_bounds, norm_type


# ============================================================================
# ENVIRONMENT INITIALIZATION
# ============================================================================

def initialize_environment(cfg: DictConfig) -> gym.Env:
    """
    Initialize the JSBSim-based flight environment using Hydra config.
    
    Args:
        cfg: Hydra DictConfig object with environment configuration (loaded by @hydra.main)
    
    Returns:
        Initialized gymnasium environment, or None if initialization fails
    """
    print("\n" + "="*60)
    print("INITIALIZING ENVIRONMENT")
    print("="*60)
    
    try:
        # Load JSBSim environment config if not already loaded
        if not hasattr(cfg.env, 'jsbsim') or cfg.env.jsbsim is None:
            cfg.env.jsbsim = OmegaConf.load('../config/env/jsbsim/noatmo.yaml')
        
        # Create environment
        env = gym.make(
            'ACBohnNoVaIErr-v0',
            cfg_env=cfg.env,
            render_mode='none'
        )
        print("  ✓ Environment created: ACBohnNoVaIErr-v0")
        print("="*60 + "\n")
        return env
    except Exception as e:
        print(f"  Error: Could not initialize environment: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return None


# ============================================================================
# MPPI ALGORITHM: TRAJECTORY ROLLOUT
# ============================================================================

@torch.no_grad()
def rollout_trajectories(
    current_state: np.ndarray,
    actions: np.ndarray,
    hybrid_model: HybridDynamicsModel,
    config: Dict,
    denorm_factors: Optional[torch.Tensor],
    min_bounds: Optional[torch.Tensor],
    norm_type: Optional[str],
    device: torch.device,
    residual_clamp: Optional[float] = None,
) -> np.ndarray:
    """
    Simulate multiple action sequences using HybridDynamicsODE — the same
    integration path used during training and test_physics_model.py.

    HybridDynamicsODE passes raw (physical) state to the physics prior and
    normalized state to the residual network, matching the training setup.
    Using HybridDynamicsModel.integrate() directly would feed normalized state
    to the physics prior, producing wrong aerodynamic forces.

    Returns:
        trajectories: (num_samples, horizon, 8) in raw physical units
                      [roll, pitch, Va(m/s), p, q, r, alpha, beta]
                      NaN for trajectories that diverged numerically.
    """
    num_samples, horizon, action_dim = actions.shape

    # Extract the 8 model state dims from the 14-dim env observation
    state_indices = [0, 1, 2, 3, 4, 5, 8, 9]
    current_state_used = current_state[state_indices].copy()
    current_state_used[2] /= 3.6   # km/h → m/s

    # Start in RAW physical space — HybridDynamicsODE integrates in raw space
    states_raw = torch.tensor(current_state_used, dtype=torch.float32, device=device)
    states_raw = states_raw.unsqueeze(0).expand(num_samples, -1).clone()

    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    dt = config['integration']['dt']

    # HybridDynamicsODE: physics_prior(state_RAW) + residual(state_NORM) * scale
    ode_module = HybridDynamicsODE(
        hybrid_model, device,
        denorm_factors=denorm_factors, min_bounds=min_bounds, norm_type=norm_type,
        residual_clamp=residual_clamp,
    )

    trajectories = torch.empty(num_samples, horizon, states_raw.shape[1], device=device)
    invalid = torch.zeros(num_samples, dtype=torch.bool, device=device)

    # Manual RK4 — avoids torchdiffeq overhead (solver setup, tensor allocation,
    # interpolation) for each single-step integration. Same math, ~5x faster.
    half_dt  = dt * 0.5
    sixth_dt = dt / 6.0

    for step_idx in range(horizon):
        action = actions_tensor[:, step_idx, :]
        ode_module.set_action(action)

        k1 = ode_module(0.0, states_raw)
        k2 = ode_module(0.0, states_raw + half_dt * k1)
        k3 = ode_module(0.0, states_raw + half_dt * k2)
        k4 = ode_module(0.0, states_raw + dt * k3)
        states_raw = states_raw + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        new_invalid = ~torch.isfinite(states_raw).all(dim=-1)
        invalid |= new_invalid
        if invalid.any():
            states_raw = torch.where(invalid.unsqueeze(-1), torch.zeros_like(states_raw), states_raw)

        trajectories[:, step_idx, :] = states_raw

    if invalid.any():
        trajectories[invalid] = float('nan')

    return trajectories.cpu().numpy()


# ============================================================================
# MPPI CONTROLLER (class-based, with warm-starting)
# ============================================================================

class MPPIController:
    """
    MPPI controller with warm-starting and correct single-temperature weighting.

    Standard MPPI formula:
        w_k = exp(-(S_k - min_S) / temperature)
        u* = Σ w_k * u_k / Σ w_k

    where S_k is the trajectory cost and temperature (λ) controls the
    sharpness of the distribution (lower = more greedy, higher = more uniform).

    Warm-starting: the nominal action sequence is shifted forward each step and
    noise is sampled around it, giving temporal coherence and faster convergence.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int,
        temperature: float,
        noise_std: float,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.temperature = temperature  # λ — do NOT also divide costs by this elsewhere
        self.noise_std = noise_std
        # Nominal action sequence, shifted forward each step (warm-start)
        self.mean_actions = np.zeros((horizon, action_dim))

    def reset(self):
        self.mean_actions = np.zeros((self.horizon, self.action_dim))
        self.mean_actions[:, 2] = 0.3  # throttle warm-start near training data mean

    def optimize(
        self,
        current_state: np.ndarray,
        target_roll: float,
        target_pitch: float,
        hybrid_model: HybridDynamicsModel,
        config: Dict,
        denorm_factors: Optional[torch.Tensor],
        min_bounds: Optional[torch.Tensor],
        norm_type: Optional[str],
        device: torch.device,
        residual_clamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        t_start = time.time()

        # --- Sample action sequences ---
        # Mix exploitative (Gaussian around warm-started mean) and purely random
        # samples. Without random samples, once the mean drifts to a boundary
        # (e.g. aileron=1.0), Gaussian noise is one-sided and the controller
        # gets stuck unable to explore the opposite direction.
        n_random = self.num_samples // 4
        n_noise  = self.num_samples - n_random

        noise = np.random.normal(
            0, self.noise_std,
            (n_noise, self.horizon, self.action_dim)
        )
        exploit_actions = np.clip(self.mean_actions[None] + noise, -1.0, 1.0)
        explore_actions = np.random.uniform(-1.0, 1.0, (n_random, self.horizon, self.action_dim))
        explore_actions[:, :, 2] = np.random.uniform(0.0, 1.0, (n_random, self.horizon))
        sampled_actions = np.concatenate([exploit_actions, explore_actions], axis=0)

        # Throttle lives in [0, 1] (aileron/elevator are [-1, 1])
        sampled_actions[:, :, 2] = np.clip(sampled_actions[:, :, 2], 0.0, 1.0)

        # --- Roll out all samples through the dynamics model ---
        t_rollout = time.time()
        trajectories = rollout_trajectories(
            current_state, sampled_actions, hybrid_model,
            config, denorm_factors, min_bounds, norm_type, device,
            residual_clamp=residual_clamp,
        )
        time_rollout = time.time() - t_rollout

        # --- Compute costs: roll/pitch tracking only ---
        target_roll_rad  = np.deg2rad(target_roll)
        target_pitch_rad = np.deg2rad(target_pitch)
        roll_errors  = np.abs(trajectories[:, :, 0] - target_roll_rad)
        pitch_errors = np.abs(trajectories[:, :, 1] - target_pitch_rad)
        costs = np.sum(roll_errors + pitch_errors, axis=1)

        # --- MPPI weights: exp(-(cost - min_cost) / temperature) ---
        valid_mask = np.isfinite(costs)
        if not valid_mask.any():
            self._shift_mean()
            zero = np.zeros(self.action_dim)
            return zero, self._debug_info(costs, valid_mask,
                                          np.ones(self.num_samples) / self.num_samples,
                                          t_start, time_rollout)

        nan_penalty = np.nanmax(costs[valid_mask]) * 10.0
        costs_safe = np.where(valid_mask, costs, nan_penalty)

        costs_shifted = costs_safe - costs_safe.min()
        weights = np.exp(-costs_shifted / self.temperature)
        weights /= weights.sum()

        # --- Update mean action sequence (weighted sum over all samples) ---
        self.mean_actions = np.einsum('k,khd->hd', weights, sampled_actions)

        # Return the first action — aileron/elevator in [-1,1], throttle in [0,1]
        best_action = self.mean_actions[0].copy()
        best_action[:2] = np.clip(best_action[:2], -1.0, 1.0)
        best_action[2]  = np.clip(best_action[2],  0.0,  1.0)

        # Shift mean forward for next call (warm-start)
        self._shift_mean()

        return best_action, self._debug_info(costs, valid_mask, weights, t_start, time_rollout)

    def _shift_mean(self):
        """Shift nominal sequence one step forward; pad last entry with neutral values."""
        self.mean_actions = np.roll(self.mean_actions, -1, axis=0)
        self.mean_actions[-1, :2] = 0.0   # aileron/elevator neutral
        self.mean_actions[-1, 2]  = 0.3   # throttle near training mean

    def _debug_info(self, costs, valid_mask, weights, t_start, time_rollout):
        # Report costs over valid trajectories only (exclude NaN-penalized ones)
        valid_costs = costs[valid_mask]
        return {
            'num_samples': self.num_samples,
            'horizon': self.horizon,
            'temperature': self.temperature,
            'time_rollout': time_rollout,
            'time_total': time.time() - t_start,
            'min_cost': float(valid_costs.min()) if len(valid_costs) else float('nan'),
            'max_cost': float(valid_costs.max()) if len(valid_costs) else float('nan'),
            'mean_cost': float(valid_costs.mean()) if len(valid_costs) else float('nan'),
            'std_cost': float(valid_costs.std()) if len(valid_costs) else float('nan'),
            'best_weight': float(weights.max()),
            'num_valid': int(valid_mask.sum()),
        }


# ============================================================================
# ENVIRONMENT CONTROL LOOP
# ============================================================================

def run_mppi_control(
    env: gym.Env,
    hybrid_model: HybridDynamicsModel,
    config: Dict,
    denorm_factors: Optional[torch.Tensor],
    min_bounds: Optional[torch.Tensor],
    norm_type: Optional[str],
    device: torch.device,
    target_roll: float = 0.0,
    target_pitch: float = 0.0,
    max_episodes: int = 5,
    max_steps_per_episode: int = 2000,
    mppi_horizon: int = 30,
    mppi_samples: int = 64,
    mppi_temperature: float = 1.0,
    mppi_noise_std: float = 0.8,
    model_path: str = "",
    residual_clamp: Optional[float] = None,
):
    """
    Main control loop: run MPPI on the environment.

    mppi_temperature (λ): controls softness of trajectory weighting.
      Lower = greedier (good for fine tracking). Typical range: 0.01–0.5.
    mppi_noise_std: std of perturbation noise around the warm-started mean.
    """
    print("\n" + "="*60)
    print("STARTING MPPI CONTROL LOOP")
    print("="*60)
    print(f"Target: Roll={target_roll:.1f}°, Pitch={target_pitch:.1f}°")
    print(f"MPPI Config: samples={mppi_samples}, horizon={mppi_horizon}, "
          f"temperature={mppi_temperature}, noise_std={mppi_noise_std}")
    print("="*60 + "\n")

    # Override the environment's PI-controller throttle with the MPPI-computed
    # throttle. apply_action() sets aileron/elevator AND runs the PI throttle,
    # then sim.run_step() uses whatever is in sim[throttle_cmd]. By monkey-patching
    # apply_action we write our throttle AFTER the PI, before run_step().
    from fw_jsbgym.utils import jsbsim_properties as prp
    mppi_throttle_ref = [0.0]
    _original_apply = env.unwrapped.apply_action

    def _patched_apply_action(action):
        _original_apply(action)
        env.unwrapped.sim[prp.throttle_cmd] = float(np.clip(mppi_throttle_ref[0], 0.0, 1.0))

    env.unwrapped.apply_action = _patched_apply_action

    controller = MPPIController(
        horizon=mppi_horizon,
        action_dim=3,
        num_samples=mppi_samples,
        temperature=mppi_temperature,
        noise_std=mppi_noise_std,
    )

    all_roll_deg   = []  # actual angle history across all episodes
    all_pitch_deg  = []
    all_actions    = []

    for episode in range(max_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{max_episodes}")
        print(f"Target: Roll={target_roll:.1f}°, Pitch={target_pitch:.1f}°")
        print(f"{'='*60}")

        controller.reset()

        # Initialize environment
        env.unwrapped.init()
        obs, info = env.reset()

        episode_reward = 0.0
        episode_roll_errors = []
        episode_pitch_errors = []
        episode_roll_deg    = []
        episode_pitch_deg   = []
        actions_taken = []
        episode_timings = []

        # Print header for step-by-step logging
        print(f"\n{'Step':>5} | {'Roll(deg)':>10} {'Err(°)':>7} | {'Pitch(deg)':>10} {'Err(°)':>7} | "
              f"{'Action [a,e,t]':>18} | {'Min':>8} {'Mean':>8} {'Time(ms)':>9}")
        print("-" * 115)

        for step in range(max_steps_per_episode):
            t_step = time.time()
            best_action, mppi_debug = controller.optimize(
                obs,
                target_roll,
                target_pitch,
                hybrid_model,
                config,
                denorm_factors,
                min_bounds,
                norm_type,
                device,
                residual_clamp=residual_clamp,
            )
            
            # Feed MPPI throttle to the monkey-patched apply_action, then step
            mppi_throttle_ref[0] = float(best_action[2])
            obs, reward, terminated, truncated, info = env.step(best_action[:2])
            episode_reward += reward
            time_step = (time.time() - t_step) * 1000  # Convert to ms
            
            # Track errors
            roll_rad = obs[0]
            pitch_rad = obs[1]
            target_roll_rad = np.deg2rad(target_roll)
            target_pitch_rad = np.deg2rad(target_pitch)
            
            roll_error = np.rad2deg(roll_rad) - target_roll
            pitch_error = np.rad2deg(pitch_rad) - target_pitch
            
            roll_error_abs = np.abs(roll_error)
            pitch_error_abs = np.abs(pitch_error)
            
            episode_roll_errors.append(roll_error_abs)
            episode_pitch_errors.append(pitch_error_abs)
            episode_roll_deg.append(np.rad2deg(roll_rad))
            episode_pitch_deg.append(np.rad2deg(pitch_rad))
            actions_taken.append(best_action)
            episode_timings.append(time_step)
            
            # Log step information (Min = lowest cost, Mean = average cost, Time = total MPPI + env step)
            print(f"{step+1:>5} | {np.rad2deg(roll_rad):>10.2f} {roll_error_abs:>7.2f} | "
                  f"{np.rad2deg(pitch_rad):>10.2f} {pitch_error_abs:>7.2f} | "
                  f"[{best_action[0]:>6.3f},{best_action[1]:>6.3f},{best_action[2]:>6.3f}] | "
                  f"{mppi_debug['min_cost']:>8.4f} {mppi_debug['mean_cost']:>8.4f} {time_step:>9.2f}")
            
            if terminated or truncated:
                print(f"\n>>> Episode terminated at step {step + 1} (reward: {reward:.4f})")
                break
        
        # Episode statistics
        avg_roll_error  = np.mean(episode_roll_errors)
        avg_pitch_error = np.mean(episode_pitch_errors)
        avg_step_time   = np.mean(episode_timings)

        roll_conv  = compute_convergence_stats(episode_roll_deg,  target_roll,  threshold_deg=1.0)
        pitch_conv = compute_convergence_stats(episode_pitch_deg, target_pitch, threshold_deg=1.0)

        all_roll_deg.extend(episode_roll_deg)
        all_pitch_deg.extend(episode_pitch_deg)
        all_actions.extend(actions_taken)

        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*60}")
        print(f"Steps executed:        {len(episode_roll_errors)}")
        print(f"Total reward:          {episode_reward:>10.4f}")
        print(f"Avg time per step:     {avg_step_time:>10.2f} ms")
        print(f"\nRoll Tracking (target={target_roll:.1f}°, threshold=±1°):")
        print(f"  Mean error:          {avg_roll_error:>10.2f}°")
        print(f"  Std dev:             {np.std(episode_roll_errors):>10.2f}°")
        print(f"  Min error:           {np.min(episode_roll_errors):>10.2f}°")
        print(f"  Max error:           {np.max(episode_roll_errors):>10.2f}°")
        if roll_conv['convergence_step'] is not None:
            print(f"  Converged at step:   {roll_conv['convergence_step']:>10d}  "
                  f"({roll_conv['convergence_step'] * 0.01:.2f}s)")
        else:
            print(f"  Converged at step:       never (threshold ±2°)")
        print(f"  Steady-state mean err:{roll_conv['steady_mean_error']:>9.2f}°")
        print(f"  Steady-state std:    {roll_conv['steady_std']:>10.2f}°")
        print(f"\nPitch Tracking (target={target_pitch:.1f}°, threshold=±1°):")
        print(f"  Mean error:          {avg_pitch_error:>10.2f}°")
        print(f"  Std dev:             {np.std(episode_pitch_errors):>10.2f}°")
        print(f"  Min error:           {np.min(episode_pitch_errors):>10.2f}°")
        print(f"  Max error:           {np.max(episode_pitch_errors):>10.2f}°")
        if pitch_conv['convergence_step'] is not None:
            print(f"  Converged at step:   {pitch_conv['convergence_step']:>10d}  "
                  f"({pitch_conv['convergence_step'] * 0.01:.2f}s)")
        else:
            print(f"  Converged at step:       never (threshold ±2°)")
        print(f"  Steady-state mean err:{pitch_conv['steady_mean_error']:>9.2f}°")
        print(f"  Steady-state std:    {pitch_conv['steady_std']:>10.2f}°")
        
        # Action statistics
        actions_array = np.array(actions_taken)
        print(f"\nAction Statistics:")
        print(f"  Aileron  - Mean: {np.mean(actions_array[:, 0]):>7.3f}  Std: {np.std(actions_array[:, 0]):>7.3f}  " 
              f"Min: {np.min(actions_array[:, 0]):>7.3f}  Max: {np.max(actions_array[:, 0]):>7.3f}")
        print(f"  Elevator - Mean: {np.mean(actions_array[:, 1]):>7.3f}  Std: {np.std(actions_array[:, 1]):>7.3f}  " 
              f"Min: {np.min(actions_array[:, 1]):>7.3f}  Max: {np.max(actions_array[:, 1]):>7.3f}")
        print(f"  Throttle - Mean: {np.mean(actions_array[:, 2]):>7.3f}  Std: {np.std(actions_array[:, 2]):>7.3f}  " 
              f"Min: {np.min(actions_array[:, 2]):>7.3f}  Max: {np.max(actions_array[:, 2]):>7.3f}")
    
    env.close()

    # Print final summary across all episodes
    print("\n" + "="*60)
    print("MPPI CONTROL COMPLETE - FINAL SUMMARY")
    print("="*60)
    print(f"Episodes completed:    {max_episodes}")
    print(f"Model:                 {model_path}")
    print(f"Target angles:         Roll={target_roll:.1f}°, Pitch={target_pitch:.1f}°")
    print(f"MPPI Configuration:    samples={mppi_samples}, horizon={mppi_horizon}, temp={mppi_temperature}")
    print("="*60 + "\n")

    return all_roll_deg, all_pitch_deg, np.array(all_actions)


# ============================================================================
# PID-ONLY BASELINE
# ============================================================================

def run_pid_control(
    env: gym.Env,
    target_roll: float = 0.0,
    target_pitch: float = 0.0,
    max_episodes: int = 1,
    max_steps_per_episode: int = 1000,
):
    """PID-only baseline using the same targets and step count as the MPPI run."""
    print("\n" + "="*60)
    print("PID BASELINE CONTROL")
    print("="*60)
    print(f"Target: Roll={target_roll:.1f}°, Pitch={target_pitch:.1f}°")
    print("="*60 + "\n")

    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)

    from fw_jsbgym.utils import jsbsim_properties as prp

    all_roll_errors  = []
    all_pitch_errors = []
    all_roll_deg     = []
    all_pitch_deg    = []
    all_actions      = []

    for episode in range(max_episodes):
        pid_aileron = PID(
            kp=1.5, ki=0.1, kd=0.1,
            dt=env.unwrapped.fdm_dt,
            trim=TrimPoint(), limit=1.0, is_throttle=False,
        )
        pid_elevator = PID(
            kp=-2.0, ki=-0.3, kd=-0.1,
            dt=env.unwrapped.fdm_dt,
            trim=TrimPoint(), limit=1.0, is_throttle=False,
        )
        pid_aileron.set_reference(target_roll_rad)
        pid_elevator.set_reference(target_pitch_rad)

        env.unwrapped.init()
        obs, _ = env.reset()

        ep_roll_errors  = []
        ep_pitch_errors = []
        ep_roll_deg     = []
        ep_pitch_deg    = []
        ep_actions      = []

        for _ in range(max_steps_per_episode):
            roll, pitch = obs[0], obs[1]
            p_radps, q_radps = obs[3], obs[4]

            aileron_cmd, _, _  = pid_aileron.update(state=roll,  state_dot=p_radps, saturate=True, normalize=False)
            elevator_cmd, _, _ = pid_elevator.update(state=pitch, state_dot=q_radps, saturate=True, normalize=False)

            obs, _, terminated, truncated, _ = env.step(np.array([aileron_cmd, elevator_cmd]))

            throttle = env.unwrapped.sim[prp.throttle_cmd]
            ep_actions.append([aileron_cmd, elevator_cmd, throttle])
            ep_roll_deg.append(np.rad2deg(obs[0]))
            ep_pitch_deg.append(np.rad2deg(obs[1]))
            ep_roll_errors.append(abs(ep_roll_deg[-1] - target_roll))
            ep_pitch_errors.append(abs(ep_pitch_deg[-1] - target_pitch))

            if terminated or truncated:
                break

        all_roll_errors.extend(ep_roll_errors)
        all_pitch_errors.extend(ep_pitch_errors)
        all_roll_deg.extend(ep_roll_deg)
        all_pitch_deg.extend(ep_pitch_deg)
        all_actions.extend(ep_actions)

    roll_conv  = compute_convergence_stats(all_roll_deg,  target_roll,  threshold_deg=1.0)
    pitch_conv = compute_convergence_stats(all_pitch_deg, target_pitch, threshold_deg=1.0)

    print("PID BASELINE - FINAL SUMMARY")
    print("="*60)
    print(f"Target angles:   Roll={target_roll:.1f}°, Pitch={target_pitch:.1f}°")
    print(f"Steps evaluated: {len(all_roll_errors)}")
    print(f"\nRoll  (threshold ±2°):")
    print(f"  Mean error: {np.mean(all_roll_errors):.2f}°  Std: {np.std(all_roll_errors):.2f}°  Max: {np.max(all_roll_errors):.2f}°")
    if roll_conv['convergence_step'] is not None:
        print(f"  Converged at step {roll_conv['convergence_step']}  ({roll_conv['convergence_step'] * 0.01:.2f}s)")
    else:
        print(f"  Never converged (threshold ±2°)")
    print(f"  Steady-state mean error: {roll_conv['steady_mean_error']:.2f}°  |  Std: {roll_conv['steady_std']:.2f}°")
    print(f"\nPitch (threshold ±2°):")
    print(f"  Mean error: {np.mean(all_pitch_errors):.2f}°  Std: {np.std(all_pitch_errors):.2f}°  Max: {np.max(all_pitch_errors):.2f}°")
    if pitch_conv['convergence_step'] is not None:
        print(f"  Converged at step {pitch_conv['convergence_step']}  ({pitch_conv['convergence_step'] * 0.01:.2f}s)")
    else:
        print(f"  Never converged (threshold ±2°)")
    print(f"  Steady-state mean error: {pitch_conv['steady_mean_error']:.2f}°  |  Std: {pitch_conv['steady_std']:.2f}°")
    print("="*60 + "\n")

    env.close()
    return all_roll_deg, all_pitch_deg, np.array(all_actions)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function to run MPPI control with the hybrid model."""
    
    parser = argparse.ArgumentParser(description='MPPI Control with Hybrid Dynamics Model')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/final_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='training_params.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--target-roll', type=float, default=20,
                        help='Target roll angle in degrees')
    parser.add_argument('--target-pitch', type=float, default=10,
                        help='Target pitch angle in degrees')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--steps-per-episode', type=int, default=1500,
                        help='Max steps per episode')
    parser.add_argument('--mppi-samples', type=int, default=600,
                        help='Number of MPPI trajectory samples')
    parser.add_argument('--mppi-horizon', type=int, default=15,
                        help='MPPI prediction horizon (steps at 0.01s each)')
    parser.add_argument('--mppi-temperature', type=float, default=1.0,
                        help='MPPI temperature λ. Match to cost scale (costs ~4-8 rad → λ~1.0)')
    parser.add_argument('--mppi-noise-std', type=float, default=0.8,
                        help='Std of perturbation noise around warm-started mean action')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--pid', action='store_true', default=False,
                        help='Run PID-only baseline after MPPI using the same targets and steps')
    parser.add_argument('--residual-clamp', type=float, default=None,
                        help='Clamp residual network output to [-x, x] in normalized space before scaling. '
                             'Prevents OOD explosion. Try 0.5 as a starting point. Default: no clamp.')
    parser.add_argument('--plot-name', type=str, default=None,
                        help='Base name for saved plot files (e.g. "ablation_v1" → ablation_v1_tracking.png / ablation_v1_actions.png). Defaults to tracking_performance.png / action_history.png')

    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model and config
    hybrid_model, model_config, denorm_factors, min_bounds, norm_type = load_config_and_model(
        args.checkpoint,
        args.config,
        device
    )
    
    # Load hydra config for environment initialization
    # We need to manually load it since we're not using @hydra.main
    try:
        from hydra import compose, initialize_config_dir
        from pathlib import Path
        config_dir = str(Path(__file__).parent / '../config')
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='default')
    except Exception as e:
        print(f"Warning: Could not load hydra config automatically: {e}")
        # Create a minimal config structure
        cfg = OmegaConf.create({
            'env': {}
        })
    
    # Initialize environment
    env = initialize_environment(cfg)
    if env is None:
        print("Error: Could not initialize environment. Exiting.")
        return
    
    # Run MPPI control
    mppi_roll, mppi_pitch, mppi_actions = run_mppi_control(
        env,
        hybrid_model,
        model_config,
        denorm_factors,
        min_bounds,
        norm_type,
        device,
        target_roll=args.target_roll,
        target_pitch=args.target_pitch,
        max_episodes=args.episodes,
        max_steps_per_episode=args.steps_per_episode,
        mppi_horizon=args.mppi_horizon,
        mppi_samples=args.mppi_samples,
        mppi_temperature=args.mppi_temperature,
        mppi_noise_std=args.mppi_noise_std,
        model_path=args.checkpoint,
        residual_clamp=args.residual_clamp,
    )

    pid_roll, pid_pitch, pid_actions = None, None, None
    if args.pid:
        env = initialize_environment(cfg)
        pid_roll, pid_pitch, pid_actions = run_pid_control(
            env,
            target_roll=args.target_roll,
            target_pitch=args.target_pitch,
            max_episodes=args.episodes,
            max_steps_per_episode=args.steps_per_episode,
        )

    tracking_filename = f"{args.plot_name}_tracking.png" if args.plot_name else 'tracking_performance.png'
    actions_filename  = f"{args.plot_name}_actions.png"  if args.plot_name else 'action_history.png'

    plot_tracking_performance(
        target_roll=args.target_roll,
        target_pitch=args.target_pitch,
        filename=tracking_filename,
        mppi_roll=mppi_roll,
        mppi_pitch=mppi_pitch,
        pid_roll=pid_roll,
        pid_pitch=pid_pitch,
    )

    plot_action_history(
        filename=actions_filename,
        mppi_actions=mppi_actions,
        pid_actions=pid_actions,
    )


if __name__ == '__main__':
    main()
