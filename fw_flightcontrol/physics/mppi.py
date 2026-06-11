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
from fw_flightcontrol.physics.utils import load_config, clean_state_dict_for_compilation


def load_config_and_model(
    model_path: str,
    config_path: str,
    device: torch.device,
    with_prior: bool = True,
    with_residual: bool = True,
) -> Tuple[HybridDynamicsModel, Dict]:
    """Load a hybrid dynamics model from a training checkpoint.

    Norm parameters are read from the checkpoint and attached to the model
    as hybrid_model.norm_scale / hybrid_model.norm_offset.

    Returns:
        (hybrid_model, config)
    """
    config        = load_config(config_path)
    physics_prior = PhysicsPrior()

    raw = torch.load(model_path, map_location=device)

    if isinstance(raw, dict) and 'residual_state' in raw:
        residual_state = clean_state_dict_for_compilation(raw['residual_state'])
        saved_epoch    = raw.get('epoch', '?')
        saved_lambda   = raw.get('lambda', '?')
        norm_scale  = torch.tensor(raw['norm_scale'],  dtype=torch.float32, device=device)
        norm_offset = torch.tensor(raw['norm_offset'], dtype=torch.float32, device=device)
        print("  ✓ Normalization parameters loaded from checkpoint")
    else:
        residual_state = clean_state_dict_for_compilation(raw)
        saved_epoch = saved_lambda = '?'
        norm_scale = norm_offset = None
        print("  ⚠ Bare state dict — no norm parameters found")

    def _infer_hidden_dims(sd):
        dims, i = [], 0
        while f'network.{i}.weight' in sd:
            if f'network.{i+2}.weight' in sd:
                dims.append(sd[f'network.{i}.weight'].shape[0])
            i += 2
        return dims

    net_config      = config['network']
    inferred_hidden = _infer_hidden_dims(residual_state)

    residual_network = PhysicsAugmented(
        state_dim=net_config['state_dim'],
        action_dim=net_config['action_dim'],
        hidden_dims=inferred_hidden or net_config['hidden_dims'],
        activation=net_config.get('activation', 'relu'),
        use_batch_norm=net_config.get('use_batch_norm', False),
    )
    residual_network.load_state_dict(residual_state)

    hybrid_model = HybridDynamicsModel(
        physics_prior=physics_prior,
        residual_network=residual_network,
        with_prior=with_prior,
        with_residual=with_residual,
    ).to(device).eval()
    hybrid_model.norm_scale  = norm_scale
    hybrid_model.norm_offset = norm_offset

    return hybrid_model, config

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


@torch.no_grad()
def rollout_trajectories(
    current_state: np.ndarray,
    actions: np.ndarray,
    hybrid_model: HybridDynamicsModel,
    config: Dict,
    device: torch.device,
    residual_clamp: Optional[float] = None,
) -> np.ndarray:
    """Roll out N action sequences through the hybrid dynamics model (manual RK4).

    Uses HybridDynamicsODE which feeds raw state to the physics prior and
    normalized state to the residual network — matching the training setup.

    Args:
        current_state: (14,) env observation
        actions:       (N, H, action_dim)

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

    ode = HybridDynamicsODE(hybrid_model, device, residual_clamp=residual_clamp)

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
    target_roll_rad  = np.deg2rad(target_roll)
    target_pitch_rad = np.deg2rad(target_pitch)
    target_va_ms     = 60.0 / 3.6

    has_nan = ~np.isfinite(trajectories[:, :, :3]).all(axis=(1, 2))
    valid = ~has_nan

    eps = 1e-6

    err_roll  = trajectories[:, :, 0] - target_roll_rad   # (N, H)
    err_pitch = trajectories[:, :, 1] - target_pitch_rad
    err_va    = trajectories[:, :, 2] - target_va_ms

    # std of raw errors across all valid samples and horizon steps (per channel)
    scale_roll  = np.std(err_roll [valid]) + eps
    scale_pitch = np.std(err_pitch[valid]) + eps
    scale_va    = np.std(err_va   [valid]) + eps

    c_roll  = np.sqrt(np.nanmean((err_roll  / scale_roll) **2, axis=1))
    c_pitch = np.sqrt(np.nanmean((err_pitch / scale_pitch)**2, axis=1))
    c_va    = np.sqrt(np.nanmean((err_va    / scale_va)   **2, axis=1))

    costs = c_roll + c_pitch + c_va
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
        min_std:     float = 0.0,
        num_elites:  int   = 64,
        momentum:    float = 0.1,
    ):
        self.horizon      = horizon
        self.action_dim   = action_dim
        self.num_samples  = num_samples
        self.temperature  = temperature  
        self.noise_std    = noise_std    
        self.min_std      = min_std      
        self.num_elites   = num_elites   
        self.momentum     = momentum     
        self.mean_actions  = np.zeros((horizon, action_dim))
        self.current_sigma = noise_std   

    def reset(self) -> None:
        self.mean_actions          = np.zeros((self.horizon, self.action_dim))
        self.mean_actions[:, 2]    = 0.3  # throttle warm-start
        self.current_sigma         = self.noise_std

    def sample_actions(self) -> np.ndarray:
        noise_ae = np.random.normal(0, self.current_sigma,       (self.num_samples, self.horizon, 2))
        noise_t  = np.random.normal(0, self.current_sigma * 0.5, (self.num_samples, self.horizon))

        actions = np.empty((self.num_samples, self.horizon, self.action_dim))
        actions[:, :, :2] = np.clip(self.mean_actions[None, :, :2] + noise_ae, -1.0, 1.0)
        actions[:, :,  2] = np.clip(self.mean_actions[None, :,  2] + noise_t,   0.0, 1.0)
        return actions

    def update(self, costs: np.ndarray, sampled_actions: np.ndarray,
               shift: bool = True) -> np.ndarray:
        valid = np.isfinite(costs)
        if not valid.any():
            best = self.mean_actions[0].copy()
            best[:2] = np.clip(best[:2], -1.0, 1.0)
            best[2]  = np.clip(best[2],   0.0, 1.0)
            if shift:
                self._shift_mean()
            return best

        costs_safe = np.where(valid, costs, np.nanmax(costs[valid]) * 10.0)

        k = min(self.num_elites, int(valid.sum()), len(costs_safe) - 1)
        elite_idx     = np.argpartition(costs_safe, k)[:k]
        elite_costs   = costs_safe[elite_idx]
        elite_actions = sampled_actions[elite_idx]

        w = np.exp(-(elite_costs - elite_costs.min()) / self.temperature)
        w /= w.sum()

        weighted_mean     = np.einsum('k,khd->hd', w, elite_actions)
        self.mean_actions = (1.0 - self.momentum) * weighted_mean + self.momentum * self.mean_actions

        diff         = elite_actions - self.mean_actions[None, :, :]
        weighted_var = np.einsum('k,khd->hd', w, diff ** 2)
        self.current_sigma = float(max(np.sqrt(np.mean(weighted_var)), self.min_std))

        best     = self.mean_actions[0].copy()
        best[:2] = np.clip(best[:2], -1.0, 1.0)
        best[2]  = np.clip(best[2],   0.0, 1.0)

        if shift:
            self._shift_mean()
        return best

    def optimize(
        self,
        current_state:  np.ndarray,
        target_roll:    float,
        target_pitch:   float,
        hybrid_model:   HybridDynamicsModel,
        config:         Dict,
        device:         torch.device,
        residual_clamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        t0 = time.time()

        sampled = self.sample_actions()

        t_rollout    = time.time()
        trajectories = rollout_trajectories(
            current_state, sampled, hybrid_model, config, device,
            residual_clamp=residual_clamp,
        )
        time_rollout = time.time() - t_rollout

        costs = compute_costs(trajectories, target_roll, target_pitch)
        best  = self.update(costs, sampled)

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
        """Shift the mean one step forward and reset σ for the next decision step."""
        self.mean_actions         = np.roll(self.mean_actions, -1, axis=0)
        self.mean_actions[-1, :2] = 0.0           # aileron/elevator neutral
        self.mean_actions[-1, 2]  = 0.3           # throttle near cruise
        self.current_sigma        = self.noise_std # reset σ to initial for next step
