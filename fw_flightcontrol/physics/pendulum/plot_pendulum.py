#!/usr/bin/env python3
"""
Plotting utilities for pendulum environment comparisons.

Generates 4 comparison plots:
1. Pendulum residual: Train vs Validation trajectory loss
2. Pendulum residual no prior: Train vs Validation trajectory loss
3. Residual vs Residual-no-prior (train losses)
4. Residual vs Residual-no-prior (validation losses)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import plotting utilities from parent physics module
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import load_tensorboard_data, get_plots_dir, FIG_DPI, FIG_FORMAT, FONT_SIZE


def plot_comparison_with_data(metrics_dict, metric_keys, fig_number,
                              title=None, ylabel=None, labels=None, 
                              save_path=None, env_name='pendulum_env'):
    """
    Plot multiple metrics on same figure for comparison.
    
    Args:
        metrics_dict (dict): Dictionary of metrics from load_tensorboard_data
        metric_keys (list): Keys to plot from metrics_dict
        fig_number (int): Figure reference number
        title (str, optional): Plot title
        ylabel (str, optional): Y-axis label
        labels (list, optional): Custom labels for each metric
        save_path (Path, optional): Where to save
        env_name (str): Environment name for folder organization
        
    Returns:
        Path: Path where figure was saved
    """
    plots_dir = get_plots_dir(env_name)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if title is None:
        title = f"{' vs '.join(labels) if labels else ' vs '.join(metric_keys)}"
    if ylabel is None:
        ylabel = 'Loss'
    if labels is None:
        labels = metric_keys
    if save_path is None:
        clean_names = [k.split('/')[-1].lower().replace(' ', '_') for k in metric_keys]
        save_name = "_vs_".join(clean_names)
        save_path = plots_dir / f"fig{fig_number:02d}_{save_name}.{FIG_FORMAT}"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for different curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot each metric
    for idx, (key, label, color) in enumerate(zip(metric_keys, labels, colors)):
        if key in metrics_dict:
            data = metrics_dict[key]
            steps = data['step'].values
            values = data['value'].values
            ax.plot(steps, values, linewidth=2.5, color=color, label=label, marker='o', 
                   markersize=4, markevery=max(1, len(steps)//20))
        else:
            print(f"Warning: Metric '{key}' not found in data")
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=FONT_SIZE + 1)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 1)
    ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_pendulum_comparisons():
    """Generate 4 comparison plots for pendulum environment."""
    
    # Path setup
    log_residual = Path('/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics/pendulum/logs/tensorboard/pendulum_residual_20260417_105407')
    log_residual_no_prior = Path('/d/tguerin/Documents/TDMPC_WORKSPACE/FW-FlightControl/fw_flightcontrol/physics/pendulum/logs/tensorboard/pendulum_residual_no_prior_20260420_102059')
    
    if not log_residual.exists():
        print(f"Error: Residual log directory not found: {log_residual}")
        return
    if not log_residual_no_prior.exists():
        print(f"Error: Residual no-prior log directory not found: {log_residual_no_prior}")
        return
    
    print("Loading TensorBoard data...")
    print(f"  Residual: {log_residual}")
    print(f"  Residual (no prior): {log_residual_no_prior}\n")
    
    metrics_residual = load_tensorboard_data(log_residual)
    metrics_residual_no_prior = load_tensorboard_data(log_residual_no_prior)
    
    print(f"Available metrics (residual): {list(metrics_residual.keys())}")
    print(f"Available metrics (no prior): {list(metrics_residual_no_prior.keys())}\n")
    
    plots_dir = get_plots_dir('pendulum_env')
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating figures in: {plots_dir}\n")
    
    saved_figures = []
    fig_counter = 1
    
    # Plot 1: Pendulum residual - Train vs Val trajectory loss
    if 'Epoch/train_loss_trajectory' in metrics_residual and 'Epoch/val_loss_trajectory' in metrics_residual:
        path = plot_comparison_with_data(
            metrics_residual,
            ['Epoch/train_loss_trajectory', 'Epoch/val_loss_trajectory'],
            fig_counter,
            title='Pendulum APHYNITY: Train vs Validation Trajectory Loss',
            ylabel='Trajectory Loss (MSE)',
            labels=['Train', 'Validation'],
            env_name='pendulum_env'
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # Plot 2: Pendulum residual no prior - Train vs Val trajectory loss
    if 'Epoch/train_loss_trajectory' in metrics_residual_no_prior and 'Epoch/val_loss_trajectory' in metrics_residual_no_prior:
        path = plot_comparison_with_data(
            metrics_residual_no_prior,
            ['Epoch/train_loss_trajectory', 'Epoch/val_loss_trajectory'],
            fig_counter,
            title='Pendulum Residual only (no prior): Train vs Validation Trajectory Loss',
            ylabel='Trajectory Loss (MSE)',
            labels=['Train', 'Validation'],
            env_name='pendulum_env'
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # Plot 3: Compare train losses (residual vs residual no prior)
    if 'Epoch/train_loss_trajectory' in metrics_residual and 'Epoch/train_loss_trajectory' in metrics_residual_no_prior:
        # Merge data for comparison
        merged_metrics = {
            'residual_train': metrics_residual['Epoch/train_loss_trajectory'],
            'residual_no_prior_train': metrics_residual_no_prior['Epoch/train_loss_trajectory']
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e']
        labels = ['APHYNITY', 'Residual only (no prior)']
        
        # Plot residual train
        data_res = metrics_residual['Epoch/train_loss_trajectory']
        ax.plot(data_res['step'].values, data_res['value'].values, linewidth=2.5, 
               color=colors[0], label=labels[0], marker='o', markersize=4)
        
        # Plot residual no prior train
        data_res_no_prior = metrics_residual_no_prior['Epoch/train_loss_trajectory']
        ax.plot(data_res_no_prior['step'].values, data_res_no_prior['value'].values, 
               linewidth=2.5, color=colors[1], label=labels[1], marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=FONT_SIZE + 1)
        ax.set_ylabel('Training Trajectory Loss (MSE)', fontsize=FONT_SIZE + 1)
        ax.set_title('Pendulum: APHYNITY vs Residual only (no prior) - Training Loss',
                    fontsize=FONT_SIZE + 2, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        save_path = plots_dir / f"fig{fig_counter:02d}_residual_vs_no_prior_train.{FIG_FORMAT}"
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        
        saved_figures.append(save_path)
        print(f"✓ Fig {fig_counter}: {save_path.name}")
        fig_counter += 1
    
    # Plot 4: Compare val losses (residual vs residual no prior)
    if 'Epoch/val_loss_trajectory' in metrics_residual and 'Epoch/val_loss_trajectory' in metrics_residual_no_prior:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e']
        labels = ['APHYNITY', 'Residual only (no prior)']
        
        # Plot residual val
        data_res = metrics_residual['Epoch/val_loss_trajectory']
        ax.plot(data_res['step'].values, data_res['value'].values, linewidth=2.5, 
               color=colors[0], label=labels[0], marker='o', markersize=4)
        
        # Plot residual no prior val
        data_res_no_prior = metrics_residual_no_prior['Epoch/val_loss_trajectory']
        ax.plot(data_res_no_prior['step'].values, data_res_no_prior['value'].values, 
               linewidth=2.5, color=colors[1], label=labels[1], marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=FONT_SIZE + 1)
        ax.set_ylabel('Validation Trajectory Loss (MSE)', fontsize=FONT_SIZE + 1)
        ax.set_title('Pendulum: APHYNITY vs Residual only (no prior) - Validation Loss',
                    fontsize=FONT_SIZE + 2, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        save_path = plots_dir / f"fig{fig_counter:02d}_residual_vs_no_prior_val.{FIG_FORMAT}"
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        
        saved_figures.append(save_path)
        print(f"✓ Fig {fig_counter}: {save_path.name}")
        fig_counter += 1
    
    print(f"\n✓ Generated {len(saved_figures)} figures")
    print(f"All figures saved to: {plots_dir}\n")
    
    return saved_figures


if __name__ == '__main__':
    plot_pendulum_comparisons()
    print("Plot generation complete!")
