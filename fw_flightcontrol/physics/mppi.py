"""
MPPI (Model Predictive Path Integral) — backend-agnostic library.

MPPIController is decoupled from any rollout backend:

    Generic usage (any backend):
        sampled = controller.sample_actions()
        costs   = your_rollout_fn(sampled, ...)   # JSBSim, hybrid model, etc.
        best    = controller.update(costs, sampled)

    Hybrid model convenience wrapper:
        best, info = controller.optimize(obs, target_roll, target_pitch,
                                         hybrid_model, ...)

This file contains no executable code — only importable functions and classes.
"""

import time
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fw_jsbgym  # noqa: F401 — registers gym envs as side effect

from fw_flightcontrol.physics.physics_prior import PhysicsPrior
from fw_flightcontrol.physics.physics_augmented import PhysicsAugmented, HybridDynamicsModel
from fw_flightcontrol.physics.training_objective import HybridDynamicsODE
from fw_flightcontrol.physics.utils import (
    load_config, get_norm_type,
    extract_bounds_from_config, compute_denorm_factors,
    clean_state_dict_for_compilation,
)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_config_and_model(
    model_path: str,
    config_path: str,
    device: torch.device,
) -> Tuple[HybridDynamicsModel, Dict, torch.Tensor, torch.Tensor, Optional[str]]:
    """Load a hybrid dynamics model from a training checkpoint.

    Returns:
        (hybrid_model, config, norm_scale, norm_offset, norm_type)
        norm_scale / norm_offset encode normalization parameters whose meaning
        depends on norm_type ('bounds_normalization', 'data_driven_normalization',
        or None for raw space).
    """
    print(f"\n{'='*60}\nLOADING MODEL AND CONFIGURATION\n{'='*60}")
    print(f"  config:     {config_path}")
    print(f"  checkpoint: {model_path}")

    config = load_config(config_path)
    physics_prior = PhysicsPrior()
    print("  ✓ Physics prior initialized (frozen)")

    raw = torch.load(model_path, map_location=device)
    norm_scale = norm_offset = None

    if isinstance(raw, dict) and 'residual_state' in raw:
        residual_state = clean_state_dict_for_compilation(raw['residual_state'])
        saved_epoch    = raw.get('epoch', '?')
        saved_lambda   = raw.get('lambda', '?')
        if 'norm_scale' in raw and 'norm_offset' in raw:
            norm_scale  = torch.tensor(raw['norm_scale'],  dtype=torch.float32, device=device)
            norm_offset = torch.tensor(raw['norm_offset'], dtype=torch.float32, device=device)
            print("  ✓ Normalization parameters loaded from checkpoint")
    else:
        residual_state = clean_state_dict_for_compilation(raw)
        saved_epoch = saved_lambda = '?'

    def _infer_hidden_dims(sd):
        dims, i = [], 0
        while f'network.{i}.weight' in sd:
            if f'network.{i+2}.weight' in sd:
                dims.append(sd[f'network.{i}.weight'].shape[0])
            i += 2
        return dims

    net_config      = config['network']
    inferred_hidden = _infer_hidden_dims(residual_state)
    if inferred_hidden != net_config['hidden_dims']:
        print(f"  ⚠ Architecture mismatch: config={net_config['hidden_dims']}, "
              f"checkpoint={inferred_hidden}. Using checkpoint.")

    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=inferred_hidden or net_config['hidden_dims'],
        activation=net_config['activation'],
        use_batch_norm=net_config['use_batch_norm'],
    )
    residual_network.load_state_dict(residual_state)
    print(f"  ✓ Residual network: {sum(p.numel() for p in residual_network.parameters()):,} params "
          f"(epoch={saved_epoch}, λ={saved_lambda})")

    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=True,
        with_residual=True,
        integration_method=config.get('integration', {}).get('method', 'rk4'),
    ).to(device).eval()
    print(f"  ✓ Hybrid model ready on {device}")

    norm_type = get_norm_type(config)
    print(f"  Normalization: {norm_type or 'none (raw space)'}")

    if norm_type is not None and norm_scale is None:
        if 'normalization_params' in config:
            p           = config['normalization_params']
            norm_scale  = torch.tensor(p['norm_scale'],  dtype=torch.float32, device=device)
            norm_offset = torch.tensor(p['norm_offset'], dtype=torch.float32, device=device)
            print("  ✓ Normalization parameters loaded from config")
        else:
            print(f"  ⚠ No normalization parameters found — disabling {norm_type}")
            norm_type = None

    print('='*60 + '\n')
    return hybrid_model, config, norm_scale, norm_offset, norm_type


# ── Environment initialization ─────────────────────────────────────────────────

def initialize_environment(cfg: DictConfig) -> gym.Env:
    """Create the JSBSim gymnasium environment from a Hydra config.

    Returns the initialized env, or None on failure.
    """
    try:
        if not hasattr(cfg.env, 'jsbsim') or cfg.env.jsbsim is None:
            cfg.env.jsbsim = OmegaConf.load(
                Path(__file__).parent.parent.parent / 'config/env/jsbsim/noatmo.yaml'
            )
        env = gym.make('ACBohnNoVaIErr-v0', cfg_env=cfg.env, render_mode='none')
        env.unwrapped.init()
        print("  ✓ Environment: ACBohnNoVaIErr-v0")
        return env
    except Exception as e:
        print(f"  ✗ Environment init failed: {e}")
        return None


# ── Hybrid model rollout ───────────────────────────────────────────────────────

@torch.no_grad()
def rollout_trajectories(
    current_state: np.ndarray,
    actions: np.ndarray,
    hybrid_model: HybridDynamicsModel,
    config: Dict,
    norm_scale: Optional[torch.Tensor],
    norm_offset: Optional[torch.Tensor],
    norm_type: Optional[str],
    device: torch.device,
    residual_clamp: Optional[float] = None,
) -> np.ndarray:
    """Roll out N action sequences through the hybrid dynamics model (manual RK4).

    Uses HybridDynamicsODE which feeds raw state to the physics prior and
    normalized state to the residual network — matching the training setup.

    Args:
        current_state: (14,) env observation
        actions:       (N, H, action_dim)
        norm_scale / norm_offset / norm_type: normalization parameters from checkpoint

    Returns:
        trajectories: (N, H, 8) in raw physical units
                      [roll, pitch, Va(m/s), p, q, r, alpha, beta]
                      NaN for trajectories that diverged.
    """
    num_samples, horizon, _ = actions.shape

    state_indices       = [0, 1, 2, 3, 4, 5, 8, 9]
    current_state_model = current_state[state_indices].copy()
    current_state_model[2] /= 3.6  # km/h → m/s

    states_raw     = torch.tensor(current_state_model, dtype=torch.float32, device=device)
    states_raw     = states_raw.unsqueeze(0).expand(num_samples, -1).clone()
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

    ode = HybridDynamicsODE(
        hybrid_model, device,
        denorm_factors=norm_scale, min_bounds=norm_offset,
        norm_type=norm_type, residual_clamp=residual_clamp,
    )

    trajectories = torch.empty(num_samples, horizon, states_raw.shape[1], device=device)
    invalid      = torch.zeros(num_samples, dtype=torch.bool, device=device)

    dt       = config['integration']['dt']
    half_dt  = dt * 0.5
    sixth_dt = dt / 6.0

    for step in range(horizon):
        ode.set_action(actions_tensor[:, step, :])
        k1 = ode(0.0, states_raw)
        k2 = ode(0.0, states_raw + half_dt * k1)
        k3 = ode(0.0, states_raw + half_dt * k2)
        k4 = ode(0.0, states_raw + dt * k3)
        states_raw = states_raw + sixth_dt * (k1 + 2.0*k2 + 2.0*k3 + k4)

        invalid |= ~torch.isfinite(states_raw).all(dim=-1)
        if invalid.any():
            states_raw = torch.where(invalid.unsqueeze(-1),
                                     torch.zeros_like(states_raw), states_raw)
        trajectories[:, step, :] = states_raw

    if invalid.any():
        trajectories[invalid] = float('nan')

    return trajectories.cpu().numpy()


# ── Cost function ──────────────────────────────────────────────────────────────

def compute_costs(
    trajectories: np.ndarray,
    target_roll:  float,
    target_pitch: float,
) -> np.ndarray:
    """Compute per-trajectory MPPI costs from a (N, H, >=3) trajectory array.

    Trajectories must have [roll_rad, pitch_rad, Va_ms, ...] in the last dim.
    NaN trajectories (diverged or terminated early) receive NaN cost.

    Normalizes each component by its std across samples so all three terms
    have equal variance and equal influence on the MPPI weighting.

    Args:
        trajectories: (N, H, >=3) — roll [rad], pitch [rad], Va [m/s]
        target_roll:  roll setpoint [°]
        target_pitch: pitch setpoint [°]

    Returns:
        costs: (N,) — NaN for invalid trajectories
    """
    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)
    target_va_ms     = 60.0 / 3.6  # 60 km/h in m/s

    c_roll  = np.sqrt(np.nanmean((trajectories[:, :, 0] - target_roll_rad)**2,  axis=1))
    c_pitch = np.sqrt(np.nanmean((trajectories[:, :, 1] - target_pitch_rad)**2, axis=1))
    c_va    = np.sqrt(np.nanmean((trajectories[:, :, 2] - target_va_ms)**2,     axis=1))

    has_nan = ~np.isfinite(trajectories[:, :, :3]).all(axis=(1, 2))

    valid_roll  = c_roll[~has_nan]
    valid_pitch = c_pitch[~has_nan]
    valid_va    = c_va[~has_nan]

    eps = 1e-6
    costs = (c_roll  / (np.std(valid_roll)  + eps) +
             c_pitch / (np.std(valid_pitch) + eps) +
             c_va    / (np.std(valid_va)    + eps))
    costs[has_nan] = np.nan
    return costs


# ── MPPI controller ────────────────────────────────────────────────────────────

class MPPIController:
    """Backend-agnostic MPPI controller with warm-starting.

    Core interface (works with any rollout source):
        sampled = controller.sample_actions()          # (N, H, action_dim)
        costs   = your_rollout_fn(sampled, ...)        # (N,) scalar per trajectory
        best    = controller.update(costs, sampled)    # (action_dim,)

    Hybrid model convenience:
        best, info = controller.optimize(obs, target_roll, target_pitch, model, ...)

    MPPI weighting:
        w_k = exp(-(S_k - min_S) / temperature)
        u*  = Σ w_k * u_k / Σ w_k
    """

    def __init__(
        self,
        horizon:     int,
        action_dim:  int,
        num_samples: int,
        temperature: float,
        noise_std:   float,
    ):
        self.horizon     = horizon
        self.action_dim  = action_dim
        self.num_samples = num_samples
        self.temperature = temperature  # λ
        self.noise_std   = noise_std
        self.mean_actions = np.zeros((horizon, action_dim))

    def reset(self) -> None:
        """Reset the warm-started mean sequence to neutral."""
        self.mean_actions         = np.zeros((self.horizon, self.action_dim))
        self.mean_actions[:, 2]   = 0.3  # throttle warm-start

    def sample_actions(self) -> np.ndarray:
        """Sample N candidate action sequences around the current mean.

        75% exploitative (Gaussian noise around mean), 25% fully random.
        Throttle uses half noise_std to avoid pile-up at the [0,1] boundary.

        Returns:
            actions: (N, H, action_dim)
        """
        n_random = self.num_samples // 4
        n_noise  = self.num_samples - n_random

        noise_ae = np.random.normal(0, self.noise_std,       (n_noise, self.horizon, 2))
        noise_t  = np.random.normal(0, self.noise_std * 0.5, (n_noise, self.horizon))

        exploit = np.empty((n_noise, self.horizon, self.action_dim))
        exploit[:, :, :2] = np.clip(self.mean_actions[None, :, :2] + noise_ae, -1.0, 1.0)
        exploit[:, :,  2] = np.clip(self.mean_actions[None, :,  2] + noise_t,   0.0, 1.0)

        explore = np.empty((n_random, self.horizon, self.action_dim))
        explore[:, :, :2] = np.random.uniform(-1.0, 1.0, (n_random, self.horizon, 2))
        explore[:, :,  2] = np.random.uniform( 0.0, 1.0, (n_random, self.horizon))

        return np.concatenate([exploit, explore], axis=0)

    def update(self, costs: np.ndarray, sampled_actions: np.ndarray) -> np.ndarray:
        """Apply MPPI weighting and return the best action.

        This is the backend-agnostic core: it does not care how costs were
        computed (JSBSim oracle, hybrid model, or any other source).

        Args:
            costs:           (N,) scalar cost per trajectory. Non-finite values
                             are replaced with a large penalty (10× max valid cost).
            sampled_actions: (N, H, action_dim)

        Returns:
            best_action: (action_dim,) first step of the weighted mean sequence,
                         clipped to valid control ranges. Returns zeros if all
                         costs are non-finite.
        """
        valid = np.isfinite(costs)
        if not valid.any():
            self._shift_mean()
            return np.zeros(self.action_dim)

        costs_safe = np.where(valid, costs, np.nanmax(costs[valid]) * 10.0)
        w = np.exp(-(costs_safe - costs_safe.min()) / self.temperature)
        w /= w.sum()

        self.mean_actions = np.einsum('k,khd->hd', w, sampled_actions)

        best       = self.mean_actions[0].copy()
        best[:2]   = np.clip(best[:2], -1.0, 1.0)
        best[2]    = np.clip(best[2],   0.0, 1.0)

        self._shift_mean()
        return best

    def optimize(
        self,
        current_state:  np.ndarray,
        target_roll:    float,
        target_pitch:   float,
        hybrid_model:   HybridDynamicsModel,
        config:         Dict,
        norm_scale:     Optional[torch.Tensor],
        norm_offset:    Optional[torch.Tensor],
        norm_type:      Optional[str],
        device:         torch.device,
        residual_clamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """One MPPI step using the hybrid dynamics model as rollout backend.

        Cost: sum of |roll_error| + |pitch_error| over the horizon (radians).

        Returns:
            best_action: (action_dim,)
            info:        dict with timing and cost statistics
        """
        t0 = time.time()

        sampled = self.sample_actions()

        t_rollout    = time.time()
        trajectories = rollout_trajectories(
            current_state, sampled, hybrid_model,
            config, norm_scale, norm_offset, norm_type, device,
            residual_clamp=residual_clamp,
        )
        time_rollout = time.time() - t_rollout

        costs = compute_costs(trajectories, target_roll, target_pitch)

        best = self.update(costs, sampled)

        valid_costs = costs[np.isfinite(costs)]
        info = {
            'num_samples':  self.num_samples,
            'horizon':      self.horizon,
            'time_rollout': time_rollout,
            'time_total':   time.time() - t0,
            'min_cost':     float(valid_costs.min())  if len(valid_costs) else float('nan'),
            'max_cost':     float(valid_costs.max())  if len(valid_costs) else float('nan'),
            'mean_cost':    float(valid_costs.mean()) if len(valid_costs) else float('nan'),
            'std_cost':     float(valid_costs.std())  if len(valid_costs) else float('nan'),
            'num_valid':    int(np.isfinite(costs).sum()),
        }
        return best, info

    def _shift_mean(self) -> None:
        """Shift the mean sequence one step forward (warm-start for next call)."""
        self.mean_actions        = np.roll(self.mean_actions, -1, axis=0)
        self.mean_actions[-1, :2] = 0.0  # aileron/elevator neutral
        self.mean_actions[-1, 2]  = 0.3  # throttle near cruise
