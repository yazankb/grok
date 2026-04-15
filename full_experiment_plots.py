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


def find_best_version(lightning_dir: str) -> str:
    """Find the version directory with the most data (most rows in metrics.csv)."""
    if not os.path.exists(lightning_dir):
        return None
    
    import pandas as pd
    
    version_dirs = [d for d in os.listdir(lightning_dir) if d.startswith("version_")]
    if not version_dirs:
        return None
    
    best_version = None
    best_rows = 0
    
    for v in version_dirs:
        metrics_file = os.path.join(lightning_dir, v, "metrics.csv")
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                if len(df) > best_rows:
                    best_rows = len(df)
                    best_version = v
            except:
                pass
    
    return best_version


def get_accuracy_columns(df) -> tuple:
    """Get step, accuracy, and loss columns from a dataframe."""
    step_col = [c for c in df.columns if 'step' in c.lower()]
    step_col = step_col[0] if step_col else df.columns[0]
    
    full_train_acc = [c for c in df.columns if 'full_train_acc' in c.lower()]
    val_acc = [c for c in df.columns if 'val_accuracy' in c.lower()]
    train_acc = [c for c in df.columns if 'train_accuracy' in c.lower() and 'full' not in c.lower()]
    
    full_train_loss = [c for c in df.columns if 'full_train_loss' in c.lower()]
    train_loss = [c for c in df.columns if 'train_loss' in c.lower() and 'full' not in c.lower()]
    val_loss = [c for c in df.columns if 'val_loss' in c.lower()]
    
    return step_col, full_train_acc, val_acc, train_acc, full_train_loss, train_loss, val_loss


def plot_specialist_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for all specialist models."""
    import pandas as pd
    
    experiment_dir = os.path.join(log_dir, experiment_name)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    plotted = 0
    for i in range(4):
        specialist_dir = os.path.join(experiment_dir, f"specialist_{i}", "lightning_logs")
        if not os.path.exists(specialist_dir):
            axes[i].text(0.5, 0.5, f"No data for specialist {i}", ha='center', va='center')
            axes[i].set_title(f"Specialist {i}")
            continue
        
        version_dir = find_best_version(specialist_dir)
        if not version_dir:
            axes[i].text(0.5, 0.5, f"No versions for specialist {i}", ha='center', va='center')
            axes[i].set_title(f"Specialist {i}")
            continue
        
        metrics_file = os.path.join(specialist_dir, version_dir, "metrics.csv")
        if not os.path.exists(metrics_file):
            axes[i].text(0.5, 0.5, f"No metrics for specialist {i}", ha='center', va='center')
            axes[i].set_title(f"Specialist {i}")
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            
            # Check if we have enough data to plot
            if len(df) <= 1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.text(0.5, 0.5, f"Insufficient training data\n(only {len(df)} row(s) logged)", 
                        ha='center', va='center', fontsize=14)
                ax.set_title("Baseline Model - Insufficient Data")
                ax.axis('off')
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=150)
                plt.close()
                return
            
            step_col, full_train_acc, val_acc, train_acc, full_train_loss, train_loss, val_loss = get_accuracy_columns(df)
            
            ax = axes[i]
            
            # Plot train accuracy
            if full_train_acc:
                ax.plot(df[step_col], df[full_train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
                plotted += 1
            elif train_acc:
                ax.plot(df[step_col], df[train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
                plotted += 1
            
            # Plot val accuracy
            if val_acc:
                ax.plot(df[step_col], df[val_acc[0]], label='Val Acc', alpha=0.8, linewidth=2)
                plotted += 1
            
            # Plot loss
            if full_train_loss:
                ax.plot(df[step_col], df[full_train_loss[0]], label='Train Loss', alpha=0.6, linestyle='--')
            elif train_loss:
                ax.plot(df[step_col], df[train_loss[0]], label='Train Loss', alpha=0.6, linestyle='--')
            
            ax.set_title(f"Specialist {i}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits appropriately
            ax.set_ylim(bottom=0)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading: {str(e)[:50]}", ha='center', va='center')
            axes[i].set_title(f"Specialist {i}")
    
    plt.suptitle("Specialist Training Curves", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_merged_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for the merged (weight averaged) model."""
    import pandas as pd
    
    experiment_dir = os.path.join(log_dir, experiment_name)
    merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
    
    if not os.path.exists(merged_dir):
        print("No merged model data found")
        return
    
    version_dir = find_best_version(merged_dir)
    if not version_dir:
        print("No version found for merged model")
        return
    
    metrics_file = os.path.join(merged_dir, version_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"No metrics file: {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    step_col, full_train_acc, val_acc, train_acc, full_train_loss, train_loss, val_loss = get_accuracy_columns(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot accuracy
    if full_train_acc:
        axes[0].plot(df[step_col], df[full_train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
    elif train_acc:
        axes[0].plot(df[step_col], df[train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
    
    if val_acc:
        axes[0].plot(df[step_col], df[val_acc[0]], label='Val Acc', alpha=0.8, linewidth=2)
    
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Plot loss
    if full_train_loss:
        axes[1].plot(df[step_col], df[full_train_loss[0]], label='Train Loss', alpha=0.8, linewidth=2)
    elif train_loss:
        axes[1].plot(df[step_col], df[train_loss[0]], label='Train Loss', alpha=0.8, linewidth=2)
    
    if val_loss:
        axes[1].plot(df[step_col], df[val_loss[0]], label='Val Loss', alpha=0.8, linewidth=2)
    
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Perplexity if available - removed as not needed
    axes[2].axis('off')
    
    plt.suptitle("Weight-Averaged Model Training Curves", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_baseline_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for the baseline model (single model trained on full data)."""
    import pandas as pd
    
    experiment_dir = os.path.join(log_dir, experiment_name)
    baseline_dir = os.path.join(experiment_dir, "baseline", "lightning_logs")
    
    if not os.path.exists(baseline_dir):
        print("No baseline model data found")
        return
    
    version_dir = find_best_version(baseline_dir)
    if not version_dir:
        print("No version found for baseline model")
        return
    
    metrics_file = os.path.join(baseline_dir, version_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"No metrics file: {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    step_col, full_train_acc, val_acc, train_acc, full_train_loss, train_loss, val_loss = get_accuracy_columns(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot accuracy
    if full_train_acc:
        axes[0].plot(df[step_col], df[full_train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
    elif train_acc:
        axes[0].plot(df[step_col], df[train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
    
    if val_acc:
        axes[0].plot(df[step_col], df[val_acc[0]], label='Val Acc', alpha=0.8, linewidth=2)
    
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Plot loss
    if full_train_loss:
        axes[1].plot(df[step_col], df[full_train_loss[0]], label='Train Loss', alpha=0.8, linewidth=2)
    elif train_loss:
        axes[1].plot(df[step_col], df[train_loss[0]], label='Train Loss', alpha=0.8, linewidth=2)
    
    if val_loss:
        axes[1].plot(df[step_col], df[val_loss[0]], label='Val Loss', alpha=0.8, linewidth=2)
    
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].axis('off')
    
    plt.suptitle("Baseline Model Training Curves (Single Model)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_distillation_curves(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot training curves for the distilled model."""
    import pandas as pd
    
    experiment_dir = os.path.join(log_dir, experiment_name)
    distill_dir = os.path.join(experiment_dir, "distillation", "lightning_logs")
    
    if not os.path.exists(distill_dir):
        # Try to find any training logs
        results_file = os.path.join(experiment_dir, "comparison_results.json")
        if not os.path.exists(results_file):
            print("No distillation data found")
            return
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if 'distill_final_metrics' in results:
            metrics = results['distill_final_metrics']
            ax.bar(['Distillation'], [metrics.get('final_student_acc', 0)], color='purple', alpha=0.7)
        
        ax.set_title("Distillation Final Student Accuracy")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()
        return
    
    version_dir = find_best_version(distill_dir)
    if not version_dir:
        print("No version found for distillation")
        return
    
    metrics_file = os.path.join(distill_dir, version_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"No metrics file: {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    step_col, full_train_acc, val_acc, train_acc, full_train_loss, train_loss, val_loss = get_accuracy_columns(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot accuracy
    if full_train_acc:
        axes[0].plot(df[step_col], df[full_train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
    elif train_acc:
        axes[0].plot(df[step_col], df[train_acc[0]], label='Train Acc', alpha=0.8, linewidth=2)
    
    if val_acc:
        axes[0].plot(df[step_col], df[val_acc[0]], label='Val Acc', alpha=0.8, linewidth=2)
    
    axes[0].set_title("Student Accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Plot loss
    if full_train_loss:
        axes[1].plot(df[step_col], df[full_train_loss[0]], label='Train Loss', alpha=0.8, linewidth=2)
    elif train_loss:
        axes[1].plot(df[step_col], df[train_loss[0]], label='Train Loss', alpha=0.8, linewidth=2)
    
    if val_loss:
        axes[1].plot(df[step_col], df[val_loss[0]], label='Val Loss', alpha=0.8, linewidth=2)
    
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].axis('off')
    
    plt.suptitle("Distillation Student Training Curves", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(log_dir: str, experiment_name: str, save_path: str = None):
    """Plot comparison of all methods."""
    import pandas as pd
    
    experiment_dir = os.path.join(log_dir, experiment_name)
    
    methods = []
    accuracies = []
    
    # Baseline
    baseline_dir = os.path.join(experiment_dir, "baseline", "lightning_logs")
    if os.path.exists(baseline_dir):
        version_dir = find_best_version(baseline_dir)
        if version_dir:
            metrics_file = os.path.join(baseline_dir, version_dir, "metrics.csv")
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                _, _, val_acc, train_acc, _, _, _ = get_accuracy_columns(df)
                acc_col = val_acc[0] if val_acc else (train_acc[0] if train_acc else None)
                if acc_col:
                    methods.append('Baseline')
                    accuracies.append(float(df[acc_col].iloc[-1]))
    
    # Weight averaged model
    merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
    if os.path.exists(merged_dir):
        version_dir = find_best_version(merged_dir)
        if version_dir:
            metrics_file = os.path.join(merged_dir, version_dir, "metrics.csv")
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                _, _, val_acc, train_acc, _, _, _ = get_accuracy_columns(df)
                acc_col = val_acc[0] if val_acc else (train_acc[0] if train_acc else None)
                if acc_col:
                    methods.append('Weight Avg')
                    accuracies.append(float(df[acc_col].iloc[-1]))
    
    # Distillation
    results_file = os.path.join(experiment_dir, "comparison_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        if 'distill_final_metrics' in results:
            methods.append('Distillation')
            accuracies.append(float(results['distill_final_metrics'].get('final_student_acc', 0)))
    
    if not methods:
        print("No comparison data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['#2E86AB', '#28A745', '#9B59B6']
    bars = ax.bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
    
    ax.set_title("Method Comparison - Final Accuracy", fontsize=14)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all(log_dir: str, experiment_name: str):
    """Generate all plots."""
    import pandas as pd
    
    experiment_dir = os.path.join(log_dir, experiment_name)
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # 1. Specialist curves (4 plots)
    plot_specialist_curves(log_dir, experiment_name, os.path.join(plots_dir, "specialists.png"))
    print("  - Specialist curves saved")
    
    # 2. Baseline curves
    plot_baseline_curves(log_dir, experiment_name, os.path.join(plots_dir, "baseline.png"))
    print("  - Baseline curves saved")
    
    # 3. Merged model curves
    plot_merged_curves(log_dir, experiment_name, os.path.join(plots_dir, "merged.png"))
    print("  - Merged model curves saved")
    
    # 4. Distillation curves
    plot_distillation_curves(log_dir, experiment_name, os.path.join(plots_dir, "distillation.png"))
    print("  - Distillation curves saved")
    
    # 5. Comparison bar chart
    plot_comparison(log_dir, experiment_name, os.path.join(plots_dir, "comparison.png"))
    print("  - Comparison plots saved")
    
    # 6. Combined summary figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Load data for summary
    methods = []
    accuracies = []
    
    # Baseline
    baseline_dir = os.path.join(experiment_dir, "baseline", "lightning_logs")
    if os.path.exists(baseline_dir):
        version_dir = find_best_version(baseline_dir)
        if version_dir:
            try:
                df = pd.read_csv(os.path.join(baseline_dir, version_dir, "metrics.csv"))
                _, _, val_acc, train_acc, _, _, _ = get_accuracy_columns(df)
                acc_col = val_acc[0] if val_acc else (train_acc[0] if train_acc else None)
                if acc_col:
                    methods.append('Baseline')
                    accuracies.append(float(df[acc_col].iloc[-1]))
            except:
                pass
    
    # Weight Avg
    merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
    if os.path.exists(merged_dir):
        version_dir = find_best_version(merged_dir)
        if version_dir:
            try:
                df = pd.read_csv(os.path.join(merged_dir, version_dir, "metrics.csv"))
                _, _, val_acc, train_acc, _, _, _ = get_accuracy_columns(df)
                acc_col = val_acc[0] if val_acc else (train_acc[0] if train_acc else None)
                if acc_col:
                    methods.append('Weight Avg')
                    accuracies.append(float(df[acc_col].iloc[-1]))
            except:
                pass
    
    # Distillation
    results_file = os.path.join(experiment_dir, "comparison_results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            if 'distill_final_metrics' in results:
                methods.append('Distillation')
                accuracies.append(float(results['distill_final_metrics'].get('final_student_acc', 0)))
        except:
            pass
    
    # Plot 1: Comparison bar chart
    colors = ['#3498DB', '#2ECC71', '#9B59B6']
    if methods:
        bars = axes[0, 0].bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
        axes[0, 0].set_title("Final Accuracy Comparison", fontsize=12)
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_ylim(0, max(accuracies) * 1.2 if accuracies else 100)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom')
    else:
        axes[0, 0].text(0.5, 0.5, "No data", ha='center', va='center')
        axes[0, 0].set_title("Final Accuracy Comparison")
    
    # Plot 2: Summary text
    summary = f"Experiment: {experiment_name}\n\n"
    summary += f"Models: 4 specialists + 1 baseline\n"
    summary += f"Methods compared: {len(methods)}\n\n"
    for m, a in zip(methods, accuracies):
        summary += f"  {m}: {a:.1f}%\n"
    
    axes[0, 1].text(0.1, 0.5, summary, fontsize=12, transform=axes[0, 1].transAxes, 
                    verticalalignment='center', family='monospace')
    axes[0, 1].axis('off')
    axes[0, 1].set_title("Summary")
    
    # Plot 3: All training curves overlay (if available)
    try:
        ax = axes[1, 0]
        legend_added = []
        
        # Load and plot merged curve
        merged_dir = os.path.join(experiment_dir, "merged_average", "lightning_logs")
        if os.path.exists(merged_dir):
            version_dir = find_best_version(merged_dir)
            if version_dir:
                df = pd.read_csv(os.path.join(merged_dir, version_dir, "metrics.csv"))
                step_col, _, val_acc, train_acc, _, _, _ = get_accuracy_columns(df)
                acc_col = val_acc[0] if val_acc else (train_acc[0] if train_acc else None)
                if acc_col:
                    ax.plot(df[step_col], df[acc_col], label='Weight Avg', alpha=0.8, linewidth=2)
        
        # Load and plot baseline
        baseline_dir = os.path.join(experiment_dir, "baseline", "lightning_logs")
        if os.path.exists(baseline_dir):
            version_dir = find_best_version(baseline_dir)
            if version_dir:
                df = pd.read_csv(os.path.join(baseline_dir, version_dir, "metrics.csv"))
                step_col, _, val_acc, train_acc, _, _, _ = get_accuracy_columns(df)
                acc_col = val_acc[0] if val_acc else (train_acc[0] if train_acc else None)
                if acc_col:
                    ax.plot(df[step_col], df[acc_col], label='Baseline', alpha=0.8, linewidth=2)
        
        ax.set_title("Training Curves Comparison")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"Error: {str(e)[:50]}", ha='center', va='center')
        axes[1, 0].set_title("Training Curves")
    
    # Plot 4: Grokking phase info
    info = "Grokking Experiment\n"
    info += "------------------\n\n"
    info += "Phase 1: Train 4 specialists\n"
    info += "        on different operators\n\n"
    info += "Phase 2: Weight averaging\n"
    info += "        merge specialist weights\n\n"
    info += "Phase 3: Knowledge distillation\n"
    info += "        train student from teachers\n\n"
    info += "Baseline: Single model trained\n"
    info += "         on full dataset"
    
    axes[1, 1].text(0.1, 0.5, info, fontsize=11, transform=axes[1, 1].transAxes,
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Methodology")
    
    plt.suptitle(f"Full Experiment Results: {experiment_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "summary.png"), dpi=150)
    plt.close()
    
    print(f"\nAll plots saved to: {plots_dir}")
    print("\nGenerated plots:")
    print("  - specialists.png (4 specialist training curves)")
    print("  - baseline.png (single model training curves)")
    print("  - merged.png (weight-averaged model curves)")
    print("  - distillation.png (student model curves)")
    print("  - comparison.png (bar chart comparison)")
    print("  - summary.png (combined summary)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="full_experiment")
    args = parser.parse_args()
    
    plot_all(args.logdir, args.experiment_name)