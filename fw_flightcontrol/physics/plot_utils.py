#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tbparse import SummaryReader
import pandas as pd


# Configuration
DEFAULT_ENV_NAME = 'uav_env'
FIG_DPI = 300
FIG_FORMAT = 'png'
FONT_SIZE = 11

def get_plots_dir(env_name=DEFAULT_ENV_NAME):
    """Get the plots directory for a specific environment."""
    return Path(__file__).parent / 'plots' / env_name
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = FONT_SIZE + 1
plt.rcParams['axes.titlesize'] = FONT_SIZE + 2
plt.rcParams['legend.fontsize'] = FONT_SIZE - 1
plt.rcParams['figure.figsize'] = (10, 6)


def load_tensorboard_data(log_dir):
    """
    Extract metrics from TensorBoard event files.
    
    Args:
        log_dir (str or Path): Directory containing TensorBoard event files
        
    Returns:
        dict: Dictionary with metric names as keys, pandas DataFrames as values
    """
    reader = SummaryReader(str(log_dir))
    df = reader.scalars
    
    # Group by metric for easier access
    metrics = {}
    if df is not None and not df.empty:
        for metric in df['tag'].unique():
            metrics[metric] = df[df['tag'] == metric].copy()
            # Sort by step for plotting
            metrics[metric] = metrics[metric].sort_values('step')
    
    return metrics


def plot_loss_curve(metric_data, metric_name, fig_number, 
                   title=None, ylabel=None, save_path=None, env_name=DEFAULT_ENV_NAME):
    """
    Create a single loss curve plot with proper formatting.
    
    Args:
        metric_data (pd.DataFrame): Data with 'step' and 'value' columns
        metric_name (str): Name of the metric (for defaults)
        fig_number (int): Figure reference number
        title (str, optional): Plot title. If None, uses metric_name
        ylabel (str, optional): Y-axis label. If None, uses metric_name
        save_path (Path, optional): Where to save. If None, uses default
        env_name (str): Environment name for folder organization
        
    Returns:
        Path: Path where figure was saved
    """
    plots_dir = get_plots_dir(env_name)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if title is None:
        title = metric_name
    if ylabel is None:
        ylabel = metric_name
    if save_path is None:
        save_path = plots_dir / f"fig{fig_number:02d}_{metric_name.lower().replace(' ', '_')}.{FIG_FORMAT}"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    steps = metric_data['step'].values
    values = metric_data['value'].values
    
    # Plot line
    ax.plot(steps, values, linewidth=2, color='#1f77b4', label=metric_name)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=FONT_SIZE + 1)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 1)
    ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_comparison_curves(metrics_dict, metric_keys, fig_number,
                          title=None, ylabel=None, labels=None, save_path=None, env_name=DEFAULT_ENV_NAME):
    """
    Plot multiple metrics on same figure for comparison (e.g., train vs val).
    
    Args:
        metrics_dict (dict): Dictionary of all metrics from load_tensorboard_data
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
        # Clean metric names for filename (remove slashes and special characters)
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
                   markersize=4, markevery=max(1, len(steps)//20))  # Show ~20 markers
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


def plot_all_epoch_metrics(log_dir, env_name=DEFAULT_ENV_NAME):
    """
    Generate all epoch-level plots from TensorBoard data.
    Creates 7 figures showing:
    1. Train Trajectory Loss
    2. Validation Trajectory Loss
    3. Train + Val Trajectory Loss (comparison)
    4. Train Regularization Loss
    5. Validation Regularization Loss
    6. Train + Val Total Loss (comparison)
    7. Learning Rate Decay
    
    Args:
        log_dir (str or Path): Directory containing TensorBoard event files
        env_name (str): Environment name for folder organization (e.g., 'uav_env', 'pendulum_env')
    """
    plots_dir = get_plots_dir(env_name)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading TensorBoard data from: {log_dir}")
    metrics = load_tensorboard_data(log_dir)
    
    print(f"\nAvailable metrics: {list(metrics.keys())}")
    print(f"Creating figures in: {plots_dir}\n")
    
    saved_figures = []
    fig_counter = 1
    
    # 1. Train Trajectory Loss
    if 'Epoch/train_loss_trajectory' in metrics:
        path = plot_loss_curve(
            metrics['Epoch/train_loss_trajectory'],
            'Train Trajectory Loss',
            fig_counter,
            title='Training Trajectory Loss per Epoch',
            ylabel='Trajectory Loss (MSE)',
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # 2. Validation Trajectory Loss
    if 'Epoch/val_loss_trajectory' in metrics:
        path = plot_loss_curve(
            metrics['Epoch/val_loss_trajectory'],
            'Validation Trajectory Loss',
            fig_counter,
            title='Validation Trajectory Loss per Epoch',
            ylabel='Trajectory Loss (MSE)',
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # 3. Train vs Val Trajectory Loss
    if 'Epoch/train_loss_trajectory' in metrics and 'Epoch/val_loss_trajectory' in metrics:
        path = plot_comparison_curves(
            metrics,
            ['Epoch/train_loss_trajectory', 'Epoch/val_loss_trajectory'],
            fig_counter,
            title='Training vs Validation Trajectory Loss',
            ylabel='Trajectory Loss (MSE)',
            labels=['Train', 'Validation'],
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # 4. Train Regularization Loss
    if 'Epoch/train_loss_regularization' in metrics:
        path = plot_loss_curve(
            metrics['Epoch/train_loss_regularization'],
            'Train Regularization Loss',
            fig_counter,
            title='Training Regularization Loss per Epoch',
            ylabel='Regularization Loss (L2)',
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # 5. Validation Regularization Loss
    if 'Epoch/val_loss_regularization' in metrics:
        path = plot_loss_curve(
            metrics['Epoch/val_loss_regularization'],
            'Validation Regularization Loss',
            fig_counter,
            title='Validation Regularization Loss per Epoch',
            ylabel='Regularization Loss (L2)',
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # 6. Train vs Val Total Loss
    if 'Epoch/train_loss_total' in metrics and 'Epoch/val_loss_total' in metrics:
        path = plot_comparison_curves(
            metrics,
            ['Epoch/train_loss_total', 'Epoch/val_loss_total'],
            fig_counter,
            title='Training vs Validation Total Loss',
            ylabel='Total Loss',
            labels=['Train', 'Validation'],
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    # 7. Learning Rate
    if 'Training/learning_rate' in metrics:
        path = plot_loss_curve(
            metrics['Training/learning_rate'],
            'Learning Rate',
            fig_counter,
            title='Learning Rate Schedule',
            ylabel='Learning Rate',
            env_name=env_name
        )
        saved_figures.append(path)
        print(f"✓ Fig {fig_counter}: {path.name}")
        fig_counter += 1
    
    print(f"\n✓ Generated {len(saved_figures)} figures")
    print(f"All figures saved to: {plots_dir}\n")
    
    return saved_figures


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate training plots from TensorBoard logs')
    parser.add_argument('log_dir', type=str, help='Path to TensorBoard log directory')
    parser.add_argument('--env-name', type=str, default='uav_env',
                        help='Environment name for output folder (default: uav_env)')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        exit(1)

    saved_figs = plot_all_epoch_metrics(log_dir, env_name=args.env_name)
    print("Plot generation complete!")
