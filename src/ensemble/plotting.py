"""Plotting utilities for ensemble visualization.

This module provides functions to create publication-ready plots for
ensemble evaluation, including ROC curves and metric comparisons.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set publication-ready style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_roc_comparison(
    individual_probs: Dict[str, np.ndarray],
    ensemble_probs: np.ndarray,
    targets: np.ndarray,
    selected_models: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot ROC curves comparing individual models and ensemble.
    
    Args:
        individual_probs: Dictionary mapping model names to probability arrays.
        ensemble_probs: Ensemble probability array (shape: n_samples x n_classes).
        targets: Ground truth labels.
        selected_models: List of model names to include (default: all).
        output_path: Path to save the figure.
        figsize: Figure size (width, height).
        
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select models to plot
    if selected_models is None:
        selected_models = list(individual_probs.keys())
    
    # Define colors for models
    colors = plt.cm.Set2(np.linspace(0, 1, len(selected_models) + 1))
    
    # Plot individual models
    for i, model_name in enumerate(selected_models):
        probs = individual_probs[model_name]
        
        # For binary classification, use positive class probability
        if probs.shape[1] == 2:
            y_score = probs[:, 1]
        else:
            y_score = probs
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(targets, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Format model name for legend
        display_name = model_name.replace("_", " ").title()
        
        # Plot ROC curve
        ax.plot(
            fpr, tpr,
            color=colors[i],
            lw=2,
            label=f'{display_name} (AUC = {roc_auc:.3f})',
            alpha=0.8,
        )
    
    # Plot ensemble
    if ensemble_probs.shape[1] == 2:
        y_score_ensemble = ensemble_probs[:, 1]
    else:
        y_score_ensemble = ensemble_probs
    
    fpr_ensemble, tpr_ensemble, _ = roc_curve(targets, y_score_ensemble)
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
    
    ax.plot(
        fpr_ensemble, tpr_ensemble,
        color='darkred',
        lw=3,
        label=f'Ensemble (AUC = {roc_auc_ensemble:.3f})',
        alpha=0.9,
    )
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3, label='Random Classifier')
    
    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison: Individual Models vs Ensemble')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    
    return fig


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot bar chart comparing metrics across models.
    
    Args:
        comparison_df: DataFrame with model comparison results.
        metrics: List of metrics to plot (default: ['Accuracy', 'F1-Score', 'ROC-AUC']).
        output_path: Path to save the figure.
        figsize: Figure size (width, height).
        
    Returns:
        Matplotlib figure object.
    """
    if metrics is None:
        metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
    
    # Filter to selected metrics
    plot_df = comparison_df[['Model'] + metrics].copy()
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    # Define colors (ensemble in different color)
    colors = ['#3498db' if model != 'Ensemble' else '#e74c3c' 
              for model in plot_df['Model']]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create bar plot
        bars = ax.bar(
            range(len(plot_df)),
            plot_df[metric],
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1,
        )
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
        
        # Formatting
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(
            [name.replace("_", " ").title() for name in plot_df['Model']],
            rotation=45,
            ha='right',
        )
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metric comparison saved to {output_path}")
    
    return fig


def plot_variance_analysis(
    variance_results: Dict,
    model_names: List[str],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot variance analysis visualization.
    
    Args:
        variance_results: Dictionary with variance analysis results.
        model_names: List of model names.
        output_path: Path to save the figure.
        figsize: Figure size (width, height).
        
    Returns:
        Matplotlib figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Agreement heatmap
    agreement_matrix = variance_results['agreement_matrix']
    
    # Format model names
    display_names = [name.replace("_", " ").title() for name in model_names]
    
    im = ax1.imshow(agreement_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_yticks(range(len(model_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.set_yticklabels(display_names)
    ax1.set_title('Model Prediction Agreement')
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = ax1.text(
                j, i, f'{agreement_matrix[i, j]:.2f}',
                ha="center", va="center",
                color="white" if agreement_matrix[i, j] > 0.5 else "black",
                fontsize=8,
            )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Agreement Rate', rotation=270, labelpad=15)
    
    # Plot 2: Accuracy distribution
    individual_accuracies = []
    for name in model_names:
        # Extract accuracy from variance results if available
        # Otherwise use mean
        individual_accuracies.append(variance_results['mean_accuracy'])
    
    # Add ensemble accuracy
    model_labels = display_names + ['Ensemble']
    accuracies = individual_accuracies + [variance_results['ensemble_accuracy']]
    colors_acc = ['#3498db'] * len(model_names) + ['#e74c3c']
    
    bars = ax2.bar(range(len(model_labels)), accuracies, color=colors_acc, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    
    # Add mean and std lines
    mean_acc = variance_results['mean_accuracy']
    std_acc = variance_results['std_accuracy']
    ax2.axhline(mean_acc, color='green', linestyle='--', alpha=0.5, label=f'Mean: {mean_acc:.3f}')
    ax2.axhline(mean_acc + std_acc, color='orange', linestyle=':', alpha=0.5, label=f'+1 STD')
    ax2.axhline(mean_acc - std_acc, color='orange', linestyle=':', alpha=0.5, label=f'-1 STD')
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(range(len(model_labels)))
    ax2.set_xticklabels(model_labels, rotation=45, ha='right')
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Variance analysis saved to {output_path}")
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """Plot confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array.
        class_names: Names of classes (default: ['Control', 'ASD']).
        output_path: Path to save the figure.
        figsize: Figure size (width, height).
        
    Returns:
        Matplotlib figure object.
    """
    if class_names is None:
        class_names = ['Control', 'ASD']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations with both counts and percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = confusion_matrix[i, j]
            percentage = cm_normalized[i, j] * 100
            text = ax.text(
                j, i,
                f'{count}\n({percentage:.1f}%)',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
                fontsize=10,
            )
    
    # Labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Ensemble Confusion Matrix')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    return fig


def create_all_plots(
    results: Dict,
    model_names: List[str],
    output_dir: Path,
):
    """Create all visualization plots for ensemble evaluation.
    
    Args:
        results: Dictionary with all evaluation results.
        model_names: List of model names.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualization plots...")
    
    # Extract data
    predictions = results.get("predictions", {})
    individual_probs = predictions.get("individual_probs", {})
    ensemble_probs = predictions.get("ensemble_probs")
    targets = predictions.get("targets")
    
    # Plot 1: ROC comparison
    if individual_probs and ensemble_probs is not None and targets is not None:
        plot_roc_comparison(
            individual_probs=individual_probs,
            ensemble_probs=ensemble_probs,
            targets=targets,
            selected_models=model_names,
            output_path=output_dir / "roc_comparison.png",
        )
    
    # Plot 2: Metric comparison
    if "comparison" in results:
        plot_metric_comparison(
            comparison_df=results["comparison"],
            output_path=output_dir / "metric_comparison.png",
        )
    
    # Plot 3: Variance analysis
    if "variance_analysis" in results:
        plot_variance_analysis(
            variance_results=results["variance_analysis"],
            model_names=model_names,
            output_path=output_dir / "variance_analysis.png",
        )
    
    # Plot 4: Confusion matrix
    if "ensemble" in results and "confusion_matrix" in results["ensemble"]:
        plot_confusion_matrix(
            confusion_matrix=results["ensemble"]["confusion_matrix"],
            output_path=output_dir / "confusion_matrix.png",
        )
    
    print(f"All plots saved to {output_dir}")
