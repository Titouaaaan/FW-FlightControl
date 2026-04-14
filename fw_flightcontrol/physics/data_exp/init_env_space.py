"""
Initialization Environment Space Analysis
===========================================

Study how the observation space at state 0 (initialization) varies across many seeds.
This is valuable information since the init space affects agent performance significantly.

The script collects initial observations across multiple seeds and analyzes:
- Statistical properties (mean, std, min, max per dimension)
- Distribution characteristics (range, quartiles)
- Variance across seeds
- State variable information

Usage:
    python init_env_space.py
    python init_env_space.py --seeds 100 --env-config constwind
    python init_env_space.py --seeds 50 --seed-range 0:50
"""

import numpy as np
import gymnasium as gym
import fw_jsbgym
import hydra
from omegaconf import DictConfig, OmegaConf
import sys


def create_env(cfg_env, env_id='ACBohnNoVaIErr-v0', jsbsim_config='constwind'):
    """Create and return a gym environment."""
    # Load jsbsim config into cfg_env
    cfg_env.jsbsim = OmegaConf.load(f'config/env/jsbsim/{jsbsim_config}.yaml')
    
    # Create environment
    env = gym.make(
        env_id,
        cfg_env=cfg_env,
        render_mode='none'
    )
    return env


def collect_init_observations(cfg_env, num_seeds=50, env_id='ACBohnNoVaIErr-v0', 
                              jsbsim_config='constwind', start_seed=0):
    """
    Collect initial observations (state 0 and state 1 after zero action) across multiple seeds.
    
    Args:
        cfg_env: Environment configuration
        num_seeds: Number of different seeds to try
        env_id: Gymnasium environment ID
        jsbsim_config: JSBSim configuration file
        start_seed: Starting seed value
        
    Returns:
        Tuple of (init_obs, step1_obs): Arrays of shape (num_seeds, obs_dim)
    """
    print(f"\n{'='*80}")
    print(f"COLLECTING OBSERVATIONS ACROSS {num_seeds} SEEDS")
    print(f"{'='*80}")
    print(f"Environment: {env_id}")
    print(f"JSBSim Config: {jsbsim_config}")
    print(f"Seed Range: {start_seed} to {start_seed + num_seeds - 1}")
    print(f"Collecting: State 0 (initialization) and State 1 (after zero action)")
    print()
    
    init_observations = []
    step1_observations = []
    
    for i, seed in enumerate(range(start_seed, start_seed + num_seeds)):
        try:
            # Create environment with specific seed
            env = create_env(cfg_env, env_id, jsbsim_config)
            
            # Initialize environment (required before reset)
            env.unwrapped.init()
            
            # Reset with seed (gymnasium supports seed parameter in reset)
            obs, info = env.reset(seed=seed)
            init_observations.append(obs.copy())
            
            # Take one step with zero action to see how environment evolves
            zero_action = np.array([0.0, 0.0])  # Zero aileron and elevator
            obs, reward, terminated, truncated, info = env.step(zero_action)
            step1_observations.append(obs.copy())
            
            env.close()
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Collected {i + 1}/{num_seeds} pairs of observations")
        
        except Exception as e:
            print(f"  Warning: Failed to collect observation for seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not init_observations:
        raise RuntimeError("Failed to collect any observations!")
    
    init_obs_array = np.array(init_observations)
    step1_obs_array = np.array(step1_observations)
    print(f"\nSuccessfully collected {len(init_observations)} observation pairs")
    print(f"Observation shape: {init_obs_array.shape}")
    
    return init_obs_array, step1_obs_array


def analyze_observations(init_obs, step1_obs, env_sample):
    """
    Analyze the initial observations and step 1 observations and print comprehensive statistics.
    
    Args:
        init_obs: Array of shape (num_seeds, obs_dim) with initial observations (state 0)
        step1_obs: Array of shape (num_seeds, obs_dim) with observations after one step
        env_sample: A sample environment for getting state variable names
    """
    num_seeds, obs_dim = init_obs.shape
    state_prps = env_sample.unwrapped.state_prps
    
    # Analyze initialization
    print(f"\n{'='*80}")
    print(f"INITIALIZATION SPACE ANALYSIS - STATE 0 (RESET)")
    print(f"{'='*80}\n")
    
    print(f"{'Idx':<4} {'State Variable':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Range':<12}")
    print("-" * 95)
    
    init_stats = []
    for i in range(obs_dim):
        obs_values = init_obs[:, i]
        
        mean = np.mean(obs_values)
        std = np.std(obs_values)
        minimum = np.min(obs_values)
        maximum = np.max(obs_values)
        range_val = maximum - minimum
        
        if i < len(state_prps):
            var_name = state_prps[i].get_legal_name()
        else:
            var_name = f"State_{i}"
        
        init_stats.append({
            'index': i,
            'name': var_name,
            'mean': mean,
            'std': std,
            'min': minimum,
            'max': maximum,
            'range': range_val,
            'values': obs_values
        })
        
        print(f"{i:<4} {var_name:<30} {mean:<12.6f} {std:<12.6f} {minimum:<12.6f} {maximum:<12.6f} {range_val:<12.6f}")
    
    print("-" * 95)
    
    # Now analyze step 1
    print(f"\n{'='*80}")
    print(f"TRANSITION SPACE ANALYSIS - STATE 1 (AFTER 1 STEP WITH ZERO ACTION)")
    print(f"{'='*80}\n")
    
    print(f"{'Idx':<4} {'State Variable':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Range':<12}")
    print("-" * 95)
    
    step1_stats = []
    for i in range(obs_dim):
        obs_values = step1_obs[:, i]
        
        mean = np.mean(obs_values)
        std = np.std(obs_values)
        minimum = np.min(obs_values)
        maximum = np.max(obs_values)
        range_val = maximum - minimum
        
        if i < len(state_prps):
            var_name = state_prps[i].get_legal_name()
        else:
            var_name = f"State_{i}"
        
        step1_stats.append({
            'index': i,
            'name': var_name,
            'mean': mean,
            'std': std,
            'min': minimum,
            'max': maximum,
            'range': range_val,
            'values': obs_values
        })
        
        print(f"{i:<4} {var_name:<30} {mean:<12.6f} {std:<12.6f} {minimum:<12.6f} {maximum:<12.6f} {range_val:<12.6f}")
    
    print("-" * 95)
    
    # Compare variance growth from state 0 to state 1
    print(f"\n{'='*80}")
    print(f"VARIANCE COMPARISON: STATE 0 vs STATE 1")
    print(f"{'='*80}\n")
    
    print(f"{'State Variable':<35} {'State 0 Std':<15} {'State 1 Std':<15} {'Growth Factor':<15}")
    print("-" * 80)
    
    high_growth_dims = []
    for i in range(obs_dim):
        var_name = init_stats[i]['name']
        std0 = init_stats[i]['std']
        std1 = step1_stats[i]['std']
        
        if std0 < 1e-10:
            if std1 > 1e-10:
                growth = float('inf')
            else:
                growth = 1.0
        else:
            growth = std1 / std0 if std0 > 0 else 1.0
        
        print(f"{var_name:<35} {std0:<15.8f} {std1:<15.8f} {growth:<15.4f}")
        
        if std1 > 0.01:  # Track dimensions that become variable
            high_growth_dims.append({
                'name': var_name,
                'std0': std0,
                'std1': std1,
                'growth': growth
            })
    
    print("-" * 80)
    print()
    
    return init_stats, step1_stats, high_growth_dims


def print_distributions(init_stats, step1_stats, num_bins=10):
    """
    Print ASCII distributions for high-variance dimensions.
    
    Args:
        init_stats: List of statistics dictionaries from analyze_observations (state 0)
        step1_stats: List of statistics dictionaries from analyze_observations (state 1)
        num_bins: Number of histogram bins to display
    """
    print(f"{'='*80}")
    print(f"DISTRIBUTION ANALYSIS - STATE 1 DIVERGENCE")
    print(f"{'='*80}\n")
    
    # Sort by std deviation at state 1
    sorted_stats = sorted(step1_stats, key=lambda x: x['std'], reverse=True)
    
    # Show top 5 most variable dimensions at state 1
    for stat in sorted_stats[:5]:
        if stat['std'] < 1e-6:  # Skip near-zero variance
            continue
            
        values = stat['values']
        name = stat['name']
        
        # Create histogram
        hist, bin_edges = np.histogram(values, bins=num_bins)
        bin_width = bin_edges[1] - bin_edges[0]
        
        print(f"{name} (μ={stat['mean']:.6f}, σ={stat['std']:.6f})")
        print("-" * 80)
        
        # Find max count for scaling
        max_count = np.max(hist) if np.max(hist) > 0 else 1
        
        for i, count in enumerate(hist):
            # Create bar
            bar_length = int(50 * count / max_count)
            bar = "█" * bar_length
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            percentage = 100 * count / len(values)
            print(f"  [{bin_start:>10.4f}, {bin_end:>10.4f}] | {bar:<50} | {count:>3} ({percentage:>5.1f}%)")
        
        print()


def print_initialization_stability(init_stats, step1_stats, high_growth_dims):
    """
    Assess initialization stability and divergence patterns.
    
    Args:
        init_stats: List of statistics dictionaries for state 0
        step1_stats: List of statistics dictionaries for state 1
        high_growth_dims: List of dimensions with significant variance growth
    """
    print(f"{'='*80}")
    print(f"INITIALIZATION STABILITY & DIVERGENCE ASSESSMENT")
    print(f"{'='*80}\n")
    
    # Classify stability at state 0
    low_var_0 = [s for s in init_stats if s['std'] < 0.01]
    medium_var_0 = [s for s in init_stats if 0.01 <= s['std'] < 0.1]
    high_var_0 = [s for s in init_stats if s['std'] >= 0.1]
    
    print("STATE 0 (At Reset) Stability Classification:")
    print(f"  Low Variance (std < 0.01):        {len(low_var_0):>3} dimensions")
    print(f"  Medium Variance (0.01 ≤ std < 0.1): {len(medium_var_0):>3} dimensions")
    print(f"  High Variance (std ≥ 0.1):        {len(high_var_0):>3} dimensions")
    print()
    
    if len(low_var_0) == len(init_stats):
        print("✓ STATE 0 IS COMPLETELY STABLE: All dimensions initialized identically across seeds")
        print("  This is the expected behavior for gym.reset() with deterministic trimmed flight.")
        print()
    
    # Classify variance at state 1
    low_var_1 = [s for s in step1_stats if s['std'] < 0.01]
    medium_var_1 = [s for s in step1_stats if 0.01 <= s['std'] < 0.1]
    high_var_1 = [s for s in step1_stats if s['std'] >= 0.1]
    
    print("STATE 1 (After 1 Step) Variance Classification:")
    print(f"  Low Variance (std < 0.01):        {len(low_var_1):>3} dimensions")
    print(f"  Medium Variance (0.01 ≤ std < 0.1): {len(medium_var_1):>3} dimensions")
    print(f"  High Variance (std ≥ 0.1):        {len(high_var_1):>3} dimensions")
    print()
    
    if high_growth_dims:
        print("⚠️  DIVERGENCE DETECTED (Dimensions becoming variable):")
        for dim in sorted(high_growth_dims, key=lambda x: x['std1'], reverse=True)[:10]:
            print(f"  • {dim['name']:<40} σ₀={dim['std0']:.2e}, σ₁={dim['std1']:.6f}")
        print()
        print("💡 Interpretation:")
        print("  - The aircraft initializes at a fixed trim point (deterministic state 0)")
        print("  - Different wind seeds cause the state to diverge rapidly in the first step")
        print("  - This means wind disturbances affect the dynamics immediately")
        print("  - Agents must be robust to seed-dependent wind effects from step 1 onward")
        print()
    else:
        print("✓ NO SIGNIFICANT DIVERGENCE: State remains stable through first step")
        print()
    
    print(f"{'='*80}")


@hydra.main(config_name='default', config_path='config', version_base=None)
def main(cfg: DictConfig):
    """Main function with hydra config. Use env_config and num_seeds overrides."""
    # Parameters can be overridden via command line:
    # python init_env_space.py +num_seeds=100 +env_config=gustsonly +num_bins=15
    
    num_seeds = cfg.get('num_seeds', 50)
    env_config = cfg.get('env_config', 'constwind')
    num_bins = cfg.get('num_bins', 10)
    
    try:
        # Collect observations (both state 0 and state 1)
        init_obs, step1_obs = collect_init_observations(
            cfg.env,
            num_seeds=num_seeds,
            jsbsim_config=env_config
        )
        
        # Analyze
        sample_env = create_env(cfg.env, jsbsim_config=env_config)
        init_stats, step1_stats, high_growth_dims = analyze_observations(init_obs, step1_obs, sample_env)
        sample_env.close()
        
        # Distributions
        print_distributions(init_stats, step1_stats, num_bins=num_bins)
        
        # Assessment
        print_initialization_stability(init_stats, step1_stats, high_growth_dims)
        
        print(f"{'='*80}")
        print("✓ Analysis complete!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
