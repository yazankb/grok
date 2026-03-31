#!/usr/bin/env python
"""
Plotting functions for full experiment results.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_specialist_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for all specialist models."""
    experiment_dir = os.path.join(log_dir, experiment_name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(4):
        specialist_dir = os.path.join(experiment_dir, f"specialist_{i}", "lightning_logs")
        if not os.path.exists(specialist_dir):
            continue
        
        version_dirs = [d for d in os.listdir(specialist_dir) if d.startswith("version_")]
        if not version_dirs:
            continue
        
        version_dir = os.path.join(specialist_dir, version_dirs[0])
        metrics_file = os.path.join(version_dir, "metrics.csv")
        
        if os.path.exists(metrics_file):
            import pandas as pd
            df = pd.read_csv(metrics_file)
            
            # Find step and accuracy columns
            step_col = [c for c in df.columns if 'step' in c.lower()][0] if any('step' in c.lower() for c in df.columns) else df.columns[0]
            val_acc_col = [c for c in df.columns if 'val_accuracy' in c.lower()]
            train_acc_col = [c for c in df.columns if 'train_accuracy' in c.lower()]
            loss_col = [c for c in df.columns if 'val_loss' in c.lower() and 'partial' not in c.lower()]
            
            ax = axes[i]
            
            if val_acc_col:
                ax.plot(df[step_col], df[val_acc_col], label='Val Acc', alpha=0.7)
            if train_acc_col:
                ax.plot(df[step_col], df[train_acc_col], label='Train Acc', alpha=0.7)
            if loss_col:
                ax.plot(df[step_col], df[loss_col], label='Loss', alpha=0.7)
            
            ax.set_title(f"Specialist {i}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle("Specialist Training Curves")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_merged_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for the merged (weight averaged) model."""
    experiment_dir = os.path.join(log_dir, experiment_name)
    merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
    
    if not os.path.exists(merged_dir):
        print("No merged model data found")
        return
    
    version_dirs = [d for d in os.listdir(merged_dir) if d.startswith("version_")]
    if not version_dirs:
        return
    
    version_dir = os.path.join(merged_dir, version_dirs[-1])  # Latest version
    metrics_file = os.path.join(version_dir, "metrics.csv")
    
    if not os.path.exists(metrics_file):
        return
    
    import pandas as pd
    df = pd.read_csv(metrics_file)
    
    step_col = [c for c in df.columns if 'step' in c.lower()][0] if any('step' in c.lower() for c in df.columns) else df.columns[0]
    val_acc_col = [c for c in df.columns if 'val_accuracy' in c.lower()]
    train_acc_col = [c for c in df.columns if 'train_accuracy' in c.lower()]
    loss_col = [c for c in df.columns if 'val_loss' in c.lower() and 'partial' not in c.lower()]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    if val_acc_col:
        axes[0].plot(df[step_col], df[val_acc_col])
        axes[0].set_title("Validation Accuracy")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].grid(True, alpha=0.3)
    
    if train_acc_col:
        axes[1].plot(df[step_col], df[train_acc_col])
        axes[1].set_title("Training Accuracy")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(True, alpha=0.3)
    
    if loss_col:
        axes[2].plot(df[step_col], df[loss_col])
        axes[2].set_title("Validation Loss")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Loss")
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Weight-Averaged Model Training Curves")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_distillation_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for the distilled model."""
    experiment_dir = os.path.join(log_dir, experiment_name)
    results_file = os.path.join(experiment_dir, "comparison_results.json")
    
    if not os.path.exists(results_file):
        print("No results file found")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Distillation metrics are logged during training - need to check if saved
    # For now, just plot the final results
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    methods = ['Distillation']
    if 'distill_final_metrics' in results:
        metrics = results['distill_final_metrics']
        ax.bar(methods, [metrics.get('final_student_acc', 0)], color='purple', alpha=0.7)
    
    ax.set_title("Distillation Final Student Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot comparison of all methods."""
    experiment_dir = os.path.join(log_dir, experiment_name)
    
    methods = []
    accuracies = []
    
    # Weight averaged model
    merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
    if os.path.exists(merged_dir):
        version_dirs = [d for d in os.listdir(merged_dir) if d.startswith("version_")]
        if version_dirs:
            metrics_file = os.path.join(merged_dir, version_dirs[-1], "metrics.csv")
            if os.path.exists(metrics_file):
                import pandas as pd
                df = pd.read_csv(metrics_file)
                val_acc_col = [c for c in df.columns if 'val_accuracy' in c.lower()]
                if val_acc_col:
                    methods.append('Weight Avg')
                    accuracies.append(df[val_acc_col].iloc[-1])
    
    # Baseline
    baseline_dir = os.path.join(experiment_dir, "baseline", "lightning_logs")
    if os.path.exists(baseline_dir):
        version_dirs = [d for d in os.listdir(baseline_dir) if d.startswith("version_")]
        if version_dirs:
            metrics_file = os.path.join(baseline_dir, version_dirs[-1], "metrics.csv")
            if os.path.exists(metrics_file):
                import pandas as pd
                df = pd.read_csv(metrics_file)
                val_acc_col = [c for c in df.columns if 'val_accuracy' in c.lower()]
                if val_acc_col:
                    methods.append('Baseline')
                    accuracies.append(df[val_acc_col].iloc[-1])
    
    # Distillation
    results_file = os.path.join(experiment_dir, "comparison_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        if 'distill_final_metrics' in results:
            methods.append('Distillation')
            accuracies.append(results['distill_final_metrics'].get('final_student_acc', 0))
    
    if not methods:
        print("No comparison data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['blue', 'green', 'purple']
    bars = ax.bar(methods, accuracies, color=colors[:len(methods)], alpha=0.7)
    
    ax.set_title("Method Comparison - Final Validation Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all(log_dir: str, experiment_name: str):
    """Generate all plots."""
    experiment_dir = os.path.join(log_dir, experiment_name)
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating plots...")
    
    plot_specialist_curves(log_dir, experiment_name, os.path.join(plots_dir, "specialists.png"))
    print("  - Specialist curves saved")
    
    plot_merged_curves(log_dir, experiment_name, os.path.join(plots_dir, "merged.png"))
    print("  - Merged model curves saved")
    
    plot_distillation_curves(log_dir, experiment_name, os.path.join(plots_dir, "distillation.png"))
    print("  - Distillation curves saved")
    
    plot_comparison(log_dir, experiment_name, os.path.join(plots_dir, "comparison.png"))
    print("  - Comparison plots saved")
    
    # Combined figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load and plot comparison
    try:
        methods = []
        accuracies = []
        
        merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
        if os.path.exists(merged_dir):
            version_dirs = [d for d in os.listdir(merged_dir) if d.startswith("version_")]
            if version_dirs:
                import pandas as pd
                df = pd.read_csv(os.path.join(merged_dir, version_dirs[-1], "metrics.csv"))
                val_col = [c for c in df.columns if 'val_accuracy' in c.lower()]
                if val_col:
                    methods.append('Weight Avg')
                    accuracies.append(float(df[val_col].iloc[-1]))
        
        baseline_dir = os.path.join(experiment_dir, "baseline", "lightning_logs")
        if os.path.exists(baseline_dir):
            version_dirs = [d for d in os.listdir(baseline_dir) if d.startswith("version_")]
            if version_dirs:
                import pandas as pd
                df = pd.read_csv(os.path.join(baseline_dir, version_dirs[-1], "metrics.csv"))
                val_col = [c for c in df.columns if 'val_accuracy' in c.lower()]
                if val_col:
                    methods.append('Baseline')
                    accuracies.append(float(df[val_col].iloc[-1]))
        
        results_file = os.path.join(experiment_dir, "comparison_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            if 'distill_final_metrics' in results:
                methods.append('Distillation')
                accuracies.append(float(results['distill_final_metrics'].get('final_student_acc', 0)))
        
        colors = ['#2E86AB', '#28A745', '#9B59B6']
        axes[0, 0].bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8)
        axes[0, 0].set_title("Final Validation Accuracy")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(True, alpha=0.3)
        for i, (m, a) in enumerate(zip(methods, accuracies)):
            axes[0, 0].text(i, a + 1, f'{a:.1f}%', ha='center')
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"No comparison data\n{str(e)}", ha='center', va='center')
    
    # Summary text
    summary = f"Experiment: {experiment_name}\n"
    summary += f"Models: 4 specialists\n"
    summary += f"Data: 50% train / operator /\n"
    summary += f"Steps: See results"
    axes[0, 1].text(0.1, 0.5, summary, fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].axis('off')
    axes[0, 1].set_title("Summary")
    
    # Placeholder for learning curves
    axes[1, 0].text(0.5, 0.5, "See individual plots", ha='center', va='center')
    axes[1, 0].set_title("Learning Curves")
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, "See individual plots", ha='center', va='center')
    axes[1, 1].set_title("Details")
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Full Experiment Results: {experiment_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "summary.png"), dpi=150)
    plt.close()
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="full_experiment")
    args = parser.parse_args()
    
    plot_all(args.logdir, args.experiment_name)