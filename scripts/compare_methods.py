#!/usr/bin/env python
"""
Compare results from weight averaging vs distillation experiments.

Usage:
    python scripts/compare_methods.py --exp_dir logs/multi_experiment
    
Or run after an experiment and it will auto-detect available results.
"""

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(log_dir: str) -> dict:
    """Load metrics from lightning CSV logs."""
    metrics = {}
    csv_path = os.path.join(log_dir, "lightning_logs", "version_0", "metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['step'])
        metrics = {
            'steps': df['step'].values,
            'train_acc': df['full_train_acc'].values if 'full_train_acc' in df.columns else None,
            'val_acc': df['val_accuracy'].values if 'val_accuracy' in df.columns else None,
            'train_loss': df['train_loss'].values if 'train_loss' in df.columns else None,
            'val_loss': df['val_loss'].values if 'val_loss' in df.columns else None,
        }
    return metrics


def load_distillation_metrics(exp_dir: str) -> dict:
    """Load distillation training metrics."""
    metrics_path = os.path.join(exp_dir, "distilled", "distill_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}


def plot_comparison(
    merged_metrics: dict,
    distill_metrics: dict,
    baseline_metrics: dict,
    output_path: str,
    title: str = "Weight Averaging vs Distillation"
):
    """Plot comparison of different merging methods."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Accuracy
    ax = axes[0, 0]
    if merged_metrics.get('train_acc') is not None:
        ax.plot(merged_metrics['steps'], merged_metrics['train_acc'], 
                label='Weight Avg', linewidth=2, color='blue')
    if distill_metrics.get('student_acc'):
        ax.plot(distill_metrics['steps'], distill_metrics['student_acc'],
                label='Distillation', linewidth=2, color='green')
    if baseline_metrics.get('train_acc') is not None:
        ax.plot(baseline_metrics['steps'], baseline_metrics['train_acc'],
                label='Baseline', linewidth=2, color='gray', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax = axes[0, 1]
    if merged_metrics.get('val_acc') is not None:
        ax.plot(merged_metrics['steps'], merged_metrics['val_acc'],
                label='Weight Avg', linewidth=2, color='blue')
    if distill_metrics.get('val_acc'):
        steps = distill_metrics.get('steps', [])
        val_acc = distill_metrics.get('val_acc', [])
        if len(steps) == len(val_acc):
            ax.plot(steps, val_acc, label='Distillation', linewidth=2, color='green')
    if baseline_metrics.get('val_acc') is not None:
        ax.plot(baseline_metrics['steps'], baseline_metrics['val_acc'],
                label='Baseline', linewidth=2, color='gray', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy (Key Metric)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    ax = axes[1, 0]
    if merged_metrics.get('train_loss') is not None:
        ax.semilogy(merged_metrics['steps'], merged_metrics['train_loss'],
                    label='Weight Avg', linewidth=2, color='blue')
    if distill_metrics.get('loss'):
        ax.semilogy(distill_metrics['steps'], distill_metrics['loss'],
                    label='Distillation', linewidth=2, color='green')
    if baseline_metrics.get('train_loss') is not None:
        ax.semilogy(baseline_metrics['steps'], baseline_metrics['train_loss'],
                    label='Baseline', linewidth=2, color='gray', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss (log scale)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distillation-specific metrics
    ax = axes[1, 1]
    if distill_metrics.get('soft_loss'):
        ax.plot(distill_metrics['steps'], distill_metrics['soft_loss'],
                label='Soft Loss', linewidth=2, color='green')
    if distill_metrics.get('hard_loss'):
        ax.plot(distill_metrics['steps'], distill_metrics['hard_loss'],
                label='Hard Loss', linewidth=2, color='orange')
    if distill_metrics.get('teacher_acc'):
        ax.plot(distill_metrics['steps'], distill_metrics['teacher_acc'],
                label='Teacher Acc', linewidth=2, color='red', linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Distillation Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    return fig


def plot_grokking_speedup(
    merged_metrics: dict,
    distill_metrics: dict,
    baseline_metrics: dict,
    output_path: str
):
    """Plot when each method achieves validation accuracy milestones."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    grok_steps = []
    final_accs = []
    
    # Check each method
    for name, metrics in [('Weight Avg', merged_metrics), 
                          ('Distillation', distill_metrics),
                          ('Baseline', baseline_metrics)]:
        if name == 'Distillation':
            val_acc = metrics.get('val_acc', [])
            steps = metrics.get('steps', [])[:len(val_acc)] if val_acc else []
        else:
            val_acc = metrics.get('val_acc')
            steps = metrics.get('steps') if val_acc is not None else None
        
        if val_acc is not None and len(val_acc) > 0:
            # Find when it first reaches 50% validation accuracy
            idx = np.where(np.array(val_acc) >= 50)[0]
            step = steps[idx[0]] if len(idx) > 0 else None
            methods.append(name)
            grok_steps.append(step if step else float('inf'))
            final_accs.append(val_acc[-1] if len(val_acc) > 0 else 0)
    
    # Bar chart
    x = np.arange(len(methods))
    colors = ['blue', 'green', 'gray']
    
    bars = ax.bar(x, [100 if s == float('inf') else s for s in grok_steps], 
                   color=colors[:len(methods)], alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Steps to 50% Val Accuracy')
    ax.set_title('Grokking Speed: Steps to 50% Validation Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add final accuracy labels on bars
    for i, (bar, acc) in enumerate(zip(bars, final_accs)):
        height = bar.get_height()
        label = f'Final: {acc:.1f}%' if acc > 0 else 'No grokking'
        if height == float('inf'):
            height = 100
            label = f'No grok (Final: {acc:.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grokking speed plot saved to: {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare grokking methods")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Experiment directory containing method subdirs")
    parser.add_argument("--merged_dir", type=str, default=None,
                        help="Weight averaging results directory")
    parser.add_argument("--distill_dir", type=str, default=None,
                        help="Distillation results directory")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Baseline results directory")
    parser.add_argument("--output", type=str, default="comparison_plot.png",
                        help="Output plot path")
    parser.add_argument("--title", type=str, 
                        default="Multi-Model Grokking: Weight Averaging vs Distillation",
                        help="Plot title")
    args = parser.parse_args()
    
    # Load metrics from each method
    merged_metrics = {}
    distill_metrics = {}
    baseline_metrics = {}
    
    if args.exp_dir:
        # Auto-detect from experiment directory
        merged_metrics = load_metrics(os.path.join(args.exp_dir, "merged_average"))
        distill_metrics = load_metrics(os.path.join(args.exp_dir, "distilled"))
        baseline_metrics = load_metrics(os.path.join(args.exp_dir, "baseline"))
    else:
        if args.merged_dir:
            merged_metrics = load_metrics(args.merged_dir)
        if args.distill_dir:
            distill_metrics = load_metrics(args.distill_dir)
        if args.baseline_dir:
            baseline_metrics = load_metrics(args.baseline_dir)
    
    # Plot comparison
    plot_comparison(merged_metrics, distill_metrics, baseline_metrics, args.output, args.title)
    
    # Plot grokking speedup
    speedup_path = args.output.replace(".png", "_speedup.png")
    plot_grokking_speedup(merged_metrics, distill_metrics, baseline_metrics, speedup_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, metrics in [('Weight Averaging', merged_metrics),
                          ('Distillation', distill_metrics),
                          ('Baseline', baseline_metrics)]:
        if name == 'Distillation' and not metrics:
            print(f"{name}: No data")
            continue
        val_acc = metrics.get('val_acc')
        if val_acc is not None and len(val_acc) > 0:
            max_val = max(val_acc)
            final_val = val_acc[-1] if len(val_acc) > 0 else 0
            steps = metrics.get('steps')
            # Find steps to 50%
            idx = np.where(np.array(val_acc) >= 50)[0]
            grok_step = steps[idx[0]] if len(idx) > 0 and steps is not None else "N/A"
            print(f"{name}: Max val acc = {max_val:.2f}%, Final = {final_val:.2f}%, Steps to 50% = {grok_step}")
        else:
            print(f"{name}: No validation data")


if __name__ == "__main__":
    main()
