"""
Integrated Gradients Explainability for BiLSTM SNP Classification Model

This script implements Integrated Gradients (IG) attribution analysis using Captum
for the trained BiLSTM model on the Autism SNP dataset. IG computes feature 
attributions by integrating gradients along a path from a baseline to the input.

Key Features:
- Uses Captum's IntegratedGradients implementation
- Supports multiple baseline strategies (zero, mean)
- Batch processing for memory efficiency
- Convergence delta computation for quality validation
- Generates rankings and visualizations matching SHAP output format
- Optional comparison with SHAP results

Advantages over SHAP:
- Faster computation (no background sampling needed)
- More stable gradients for deep models
- Theoretical guarantees (completeness axiom)
- Deterministic results

Usage:
    python src/integrated_gradients_explainability.py --checkpoint_path <path_to_best_ckpt>
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset

from captum.attr import IntegratedGradients

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BiLSTMWrapper(torch.nn.Module):
    """Wrapper for BiLSTM model compatible with Captum attribution methods.
    
    This wrapper ensures the model output is suitable for attribution:
    - Returns logits for the target class
    - Handles BiLSTM's internal reshaping
    """
    
    def __init__(self, model: torch.nn.Module, target_class: int = 1):
        """Initialize the wrapper.
        
        Args:
            model: The BiLSTM model (net component of LightningModule)
            target_class: Class to compute attributions for (default: 1 for case)
        """
        super().__init__()
        self.model = model
        self.target_class = target_class
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, num_features)
            
        Returns:
            Logits for target class of shape (batch,)
        """
        logits = self.model(x)  # Shape: (batch, num_classes)
        return logits[:, self.target_class]  # Return only target class logits


def load_checkpoint(checkpoint_path: str) -> Tuple[LightningModule, Dict]:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        
    Returns:
        Tuple of (loaded_model, checkpoint_dict)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Import the LitModule
    from src.models.module import LitModule
    
    # Load checkpoint with weights_only=False for Lightning checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    
    logger.info(f"Model hyperparameters: {hparams}")
    
    # Load the model
    model = LitModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    
    return model, checkpoint


def load_data_with_identifiers(
    data_file: str,
    datamodule_state: Dict,
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Load data and extract SNP identifiers, applying same preprocessing.
    
    Args:
        data_file: Path to the CSV file
        datamodule_state: State dict from the checkpoint's datamodule
        train_val_test_split: Train/val/test split ratios
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels, snp_ids)
    """
    logger.info(f"Loading data from: {data_file}")
    
    # Load CSV with SNP identifiers
    df = pd.read_csv(data_file, header=0, index_col=0, low_memory=False)
    
    # Extract labels from last row
    labels = df.iloc[-1].values
    df = df.iloc[:-1]  # Remove label row
    
    # Store SNP identifiers (row index)
    all_snp_ids = df.index.tolist()
    logger.info(f"Total SNPs in original data: {len(all_snp_ids)}")
    
    # Convert labels to numeric
    labels_numeric = np.zeros(len(labels), dtype=int)
    labels_lower = np.array([str(label).lower().strip() for label in labels])
    for i, label in enumerate(labels_lower):
        if label == 'case':
            labels_numeric[i] = 1
        elif label == 'control':
            labels_numeric[i] = 0
        else:
            try:
                labels_numeric[i] = int(label)
            except ValueError:
                raise ValueError(f"Unknown label '{labels[i]}' at index {i}")
    
    # Transpose: samples as rows, SNPs as columns
    data = df.T
    data_values = data.values.astype(float)
    data_values = np.nan_to_num(data_values, nan=0.0)
    
    # Convert to tensors
    data_tensor = torch.tensor(data_values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_numeric, dtype=torch.long)
    
    logger.info(f"Data shape: {data_tensor.shape}, Labels shape: {labels_tensor.shape}")
    
    # Apply normalization if present in checkpoint
    if 'mean' in datamodule_state and 'std' in datamodule_state:
        mean = datamodule_state['mean']
        std = datamodule_state['std']
        
        # Ensure tensors
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)
        
        # Normalize
        std = torch.where(std == 0, torch.ones_like(std), std)
        data_tensor = (data_tensor - mean) / std
        logger.info("✓ Applied normalization (z-score)")
    
    # Apply feature selection if present in checkpoint
    selected_snp_ids = all_snp_ids
    if 'selected_indices' in datamodule_state:
        selected_indices = datamodule_state['selected_indices']
        if not isinstance(selected_indices, np.ndarray):
            selected_indices = np.array(selected_indices)
        
        data_tensor = data_tensor[:, selected_indices]
        selected_snp_ids = [all_snp_ids[i] for i in selected_indices]
        logger.info(f"✓ Applied feature selection: {len(selected_snp_ids)} SNPs selected")
    
    # Split data using same seed as training
    total_size = len(data_tensor)
    train_size = int(train_val_test_split[0] * total_size)
    val_size = int(train_val_test_split[1] * total_size)
    
    # Use same random seed as training
    torch.manual_seed(42)
    indices = torch.randperm(total_size).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = data_tensor[train_indices]
    train_labels = labels_tensor[train_indices]
    test_data = data_tensor[test_indices]
    test_labels = labels_tensor[test_indices]
    
    logger.info(f"✓ Split complete: Train={len(train_data)}, Test={len(test_data)}")
    
    return train_data, train_labels, test_data, test_labels, selected_snp_ids


def compute_baseline(
    train_data: torch.Tensor,
    baseline_type: str = 'zero',
    num_samples: int = 100,
) -> torch.Tensor:
    """Compute baseline for Integrated Gradients.
    
    Args:
        train_data: Training data tensor
        baseline_type: Type of baseline ('zero', 'mean', 'random')
        num_samples: Number of samples for random baseline
        
    Returns:
        Baseline tensor of shape (1, num_features) or (num_samples, num_features)
    """
    logger.info(f"Computing baseline: {baseline_type}")
    
    if baseline_type == 'zero':
        # Zero baseline - all features set to 0
        baseline = torch.zeros(1, train_data.shape[1])
    elif baseline_type == 'mean':
        # Mean baseline - population mean for each feature
        baseline = train_data.mean(dim=0, keepdim=True)
    elif baseline_type == 'random':
        # Random sampling from training distribution
        torch.manual_seed(42)
        indices = torch.randperm(len(train_data))[:num_samples]
        baseline = train_data[indices]
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    logger.info(f"✓ Baseline computed: {baseline.shape}")
    return baseline


def compute_integrated_gradients(
    model: LightningModule,
    test_data: torch.Tensor,
    baseline: torch.Tensor,
    num_test: int = 100,
    n_steps: int = 50,
    batch_size: int = 16,
    device: str = 'cpu',
    internal_batch_size: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Integrated Gradients attributions.
    
    Args:
        model: Trained LightningModule
        test_data: Test data tensor to explain
        baseline: Baseline tensor for IG
        num_test: Number of test samples to explain
        n_steps: Number of steps for Riemann approximation
        batch_size: Batch size for processing samples
        device: Device to run computation on
        internal_batch_size: Internal batch size for IG computation
        
    Returns:
        Tuple of (attributions, deltas) as numpy arrays
    """
    logger.info("Computing Integrated Gradients attributions...")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Extract the neural network from LightningModule
    net = model.net
    
    # Sample test data
    num_test = min(num_test, len(test_data))
    torch.manual_seed(42)
    test_indices = torch.randperm(len(test_data))[:num_test]
    test_samples = test_data[test_indices]
    
    logger.info(f"Test samples: {test_samples.shape}")
    logger.info(f"Baseline: {baseline.shape}")
    logger.info(f"Integration steps: {n_steps}")
    
    # Create wrapper for the network
    wrapped_model = BiLSTMWrapper(net, target_class=1)  # Class 1 = case
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    # Initialize IntegratedGradients
    logger.info("Initializing IntegratedGradients...")
    ig = IntegratedGradients(wrapped_model)
    
    # Move baseline to device
    baseline = baseline.to(device)
    
    # Process in batches for memory efficiency
    all_attributions = []
    all_deltas = []
    
    num_batches = (len(test_samples) + batch_size - 1) // batch_size
    logger.info(f"Processing {num_batches} batches of size {batch_size}...")
    
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size].to(device)
        batch.requires_grad = True
        
        # Handle baseline broadcasting
        if baseline.shape[0] == 1:
            # Single baseline: broadcast to batch size
            batch_baseline = baseline.expand(batch.shape[0], -1)
        else:
            # Multiple baselines: use one per sample (or subsample)
            batch_baseline = baseline[:batch.shape[0]] if baseline.shape[0] >= batch.shape[0] else baseline
        
        # Compute attributions with delta (convergence metric)
        try:
            attributions, delta = ig.attribute(
                batch,
                baselines=batch_baseline,
                n_steps=n_steps,
                return_convergence_delta=True,
                internal_batch_size=internal_batch_size,
            )
            
            # Move to CPU and store
            all_attributions.append(attributions.cpu().detach())
            all_deltas.append(delta.cpu().detach())
            
            # Progress update
            current_batch = (i // batch_size) + 1
            logger.info(f"  Batch {current_batch}/{num_batches} complete | "
                       f"Mean delta: {delta.mean().item():.6f}")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            raise
    
    # Concatenate all batches
    attributions_tensor = torch.cat(all_attributions, dim=0)
    deltas_tensor = torch.cat(all_deltas, dim=0)
    
    # Convert to numpy
    attributions_np = attributions_tensor.numpy()
    deltas_np = deltas_tensor.numpy()
    
    logger.info(f"✓ IG attributions computed: {attributions_np.shape}")
    logger.info(f"✓ Convergence deltas: mean={deltas_np.mean():.6f}, "
               f"max={deltas_np.max():.6f}, std={deltas_np.std():.6f}")
    
    return attributions_np, deltas_np


def aggregate_and_rank_snps(
    attributions: np.ndarray,
    deltas: np.ndarray,
    snp_ids: List[str],
    output_path: str,
) -> pd.DataFrame:
    """Aggregate IG attributions and rank SNPs by importance.
    
    Args:
        attributions: IG attributions of shape (num_samples, num_features)
        deltas: Convergence deltas of shape (num_samples,)
        snp_ids: List of SNP identifiers
        output_path: Path to save the ranked SNP list
        
    Returns:
        DataFrame with ranked SNPs and their importance scores
    """
    logger.info("Aggregating IG attributions across samples...")
    
    # Compute mean absolute attribution for each feature
    mean_abs_attr = np.mean(np.abs(attributions), axis=0)
    
    # Create DataFrame
    snp_importance_df = pd.DataFrame({
        'SNP_ID': snp_ids,
        'Mean_Abs_IG': mean_abs_attr,
        'Mean_IG': np.mean(attributions, axis=0),
        'Std_IG': np.std(attributions, axis=0),
        'Max_Abs_IG': np.max(np.abs(attributions), axis=0),
    })
    
    # Sort by mean absolute attribution (descending)
    snp_importance_df = snp_importance_df.sort_values('Mean_Abs_IG', ascending=False)
    snp_importance_df['Rank'] = range(1, len(snp_importance_df) + 1)
    
    # Reorder columns
    snp_importance_df = snp_importance_df[['Rank', 'SNP_ID', 'Mean_Abs_IG', 'Mean_IG', 'Std_IG', 'Max_Abs_IG']]
    
    # Save to CSV
    snp_importance_df.to_csv(output_path, index=False)
    logger.info(f"✓ Ranked SNPs saved to: {output_path}")
    
    # Print top 20
    logger.info("\nTop 20 most important SNPs (by Integrated Gradients):")
    print(snp_importance_df.head(20).to_string(index=False))
    
    return snp_importance_df


def create_bar_plot(
    snp_importance_df: pd.DataFrame,
    output_path: str,
    top_k: int = 20,
):
    """Create bar plot of top K most important SNPs.
    
    Args:
        snp_importance_df: DataFrame with ranked SNPs
        output_path: Path to save the figure
        top_k: Number of top SNPs to display
    """
    logger.info(f"Creating bar plot for top {top_k} SNPs...")
    
    # Get top K SNPs
    top_snps = snp_importance_df.head(top_k)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = ax.barh(range(len(top_snps)), top_snps['Mean_Abs_IG'], color='coral')
    
    # Customize plot
    ax.set_yticks(range(len(top_snps)))
    ax.set_yticklabels(top_snps['SNP_ID'], fontsize=10)
    ax.set_xlabel('Mean Absolute Integrated Gradients', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNP Identifier', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important SNPs for Autism Classification\n(BiLSTM Model - Integrated Gradients)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_snps['Mean_Abs_IG'])):
        ax.text(value + 0.001, i, f'{value:.4f}', 
                va='center', fontsize=9, color='black')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Bar plot saved to: {output_path}")
    plt.close()


def create_heatmap(
    attributions: np.ndarray,
    test_labels: np.ndarray,
    snp_importance_df: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
    num_samples: int = 30,
):
    """Create heatmap showing per-sample IG attributions for top SNPs.
    
    Args:
        attributions: IG attributions of shape (num_samples, num_features)
        test_labels: Test labels (0=control, 1=case)
        snp_importance_df: DataFrame with ranked SNPs
        output_path: Path to save the figure
        top_k: Number of top SNPs to display
        num_samples: Number of samples to display
    """
    logger.info(f"Creating heatmap for top {top_k} SNPs across {num_samples} samples...")
    
    # Get indices of top K SNPs
    top_snp_ids = snp_importance_df.head(top_k)['SNP_ID'].tolist()
    all_snp_ids = snp_importance_df['SNP_ID'].tolist()
    top_indices = [all_snp_ids.index(snp_id) for snp_id in top_snp_ids]
    
    # Select attributions for top SNPs
    attr_top = attributions[:, top_indices]
    
    # Limit number of samples for visualization
    if attr_top.shape[0] > num_samples:
        # Sample equal number of cases and controls if possible
        case_indices = np.where(test_labels == 1)[0]
        control_indices = np.where(test_labels == 0)[0]
        
        num_cases = min(num_samples // 2, len(case_indices))
        num_controls = min(num_samples - num_cases, len(control_indices))
        
        np.random.seed(42)
        selected_indices = np.concatenate([
            np.random.choice(case_indices, num_cases, replace=False),
            np.random.choice(control_indices, num_controls, replace=False)
        ])
        
        attr_top = attr_top[selected_indices]
        test_labels_subset = test_labels[selected_indices]
    else:
        test_labels_subset = test_labels
    
    # Transpose for better visualization (SNPs as rows, samples as columns)
    attr_top = attr_top.T
    
    # Create sample labels
    sample_labels = [f"Case {i+1}" if label == 1 else f"Control {i+1}" 
                    for i, label in enumerate(test_labels_subset)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create heatmap
    sns.heatmap(
        attr_top,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'IG Attribution', 'shrink': 0.8},
        xticklabels=sample_labels,
        yticklabels=top_snp_ids if top_k <= 50 else False,
        ax=ax,
        vmin=-np.abs(attr_top).max(),
        vmax=np.abs(attr_top).max(),
    )
    
    # Customize plot
    ax.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNP Identifier', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Sample IG Attributions for Top {top_k} SNPs\n'
                 f'(BiLSTM Autism Classification Model)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    if top_k <= 50:
        plt.setp(ax.get_yticklabels(), fontsize=8)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Heatmap saved to: {output_path}")
    plt.close()


def compare_with_shap(
    ig_df: pd.DataFrame,
    shap_csv_path: str,
    output_path: str,
    top_k: int = 50,
):
    """Compare IG results with SHAP results if available.
    
    Args:
        ig_df: DataFrame with IG rankings
        shap_csv_path: Path to SHAP results CSV
        output_path: Path to save comparison figure
        top_k: Number of top SNPs to compare
    """
    if not Path(shap_csv_path).exists():
        logger.info(f"SHAP results not found at {shap_csv_path}, skipping comparison")
        return
    
    logger.info("Comparing IG and SHAP results...")
    
    # Load SHAP results
    shap_df = pd.read_csv(shap_csv_path)
    
    # Get top K SNPs from each method
    ig_top = set(ig_df.head(top_k)['SNP_ID'])
    shap_top = set(shap_df.head(top_k)['SNP_ID'])
    
    # Compute overlap
    overlap = ig_top.intersection(shap_top)
    overlap_pct = len(overlap) / top_k * 100
    
    logger.info(f"Top {top_k} SNP overlap: {len(overlap)} ({overlap_pct:.1f}%)")
    logger.info(f"Consensus SNPs: {sorted(overlap)[:10]}")
    
    # Compute rank correlation for common SNPs
    common_snps = list(ig_top.intersection(shap_top))
    if len(common_snps) > 0:
        ig_ranks = [ig_df[ig_df['SNP_ID'] == snp]['Rank'].values[0] for snp in common_snps]
        shap_ranks = [shap_df[shap_df['SNP_ID'] == snp]['Rank'].values[0] for snp in common_snps]
        
        from scipy.stats import spearmanr
        corr, pval = spearmanr(ig_ranks, shap_ranks)
        logger.info(f"Spearman rank correlation: {corr:.3f} (p={pval:.3e})")
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Venn diagram-style visualization
    ax1.text(0.3, 0.6, f'IG Only\n{len(ig_top - shap_top)}', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(0.7, 0.6, f'SHAP Only\n{len(shap_top - ig_top)}', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.4, f'Overlap\n{len(overlap)}\n({overlap_pct:.1f}%)', 
             ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title(f'Top {top_k} SNP Overlap Between Methods', fontsize=14, fontweight='bold')
    
    # Scatter plot of importance scores for top consensus SNPs
    if len(overlap) > 0:
        consensus_snps = list(overlap)[:20]  # Top 20 consensus
        ig_scores = [ig_df[ig_df['SNP_ID'] == snp]['Mean_Abs_IG'].values[0] for snp in consensus_snps]
        shap_scores = [shap_df[shap_df['SNP_ID'] == snp]['Mean_Abs_SHAP'].values[0] for snp in consensus_snps]
        
        ax2.scatter(shap_scores, ig_scores, s=100, alpha=0.6, color='purple')
        ax2.set_xlabel('SHAP Importance (Mean Abs)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('IG Importance (Mean Abs)', fontsize=12, fontweight='bold')
        ax2.set_title('Importance Score Correlation\n(Top 20 Consensus SNPs)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add diagonal reference line
        min_val = min(min(shap_scores), min(ig_scores))
        max_val = max(max(shap_scores), max(ig_scores))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect agreement')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Comparison figure saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Integrated Gradients Explainability Analysis for BiLSTM SNP Classification'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to the best model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='data/FinalizedAutismData.csv',
        help='Path to the data CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: same as checkpoint dir)'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=100,
        help='Number of test samples to explain (default: 100)'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=50,
        help='Number of integration steps (default: 50, higher=more accurate)'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='zero',
        choices=['zero', 'mean', 'random'],
        help='Baseline type for IG (default: zero)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing samples (default: 16)'
    )
    parser.add_argument(
        '--internal_batch_size',
        type=int,
        default=None,
        help='Internal batch size for IG computation (default: None=auto)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation (default: auto-detect)'
    )
    parser.add_argument(
        '--top_k_bar',
        type=int,
        default=20,
        help='Number of top SNPs for bar plot (default: 20)'
    )
    parser.add_argument(
        '--top_k_heatmap',
        type=int,
        default=100,
        help='Number of top SNPs for heatmap (default: 100)'
    )
    parser.add_argument(
        '--compare_shap',
        action='store_true',
        help='Compare with SHAP results if available'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        export_dir = os.getenv("SNP_EXPORT_DIR")
        if export_dir:
            args.output_dir = Path(export_dir) / "ig_analysis"
        else:
            checkpoint_dir = Path(args.checkpoint_path).parent.parent
            args.output_dir = checkpoint_dir / 'ig_analysis'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint_path)
    
    # Extract datamodule state
    datamodule_state = checkpoint.get('datamodule', checkpoint.get('DataModule', {}))
    
    # Load data with preprocessing
    train_data, train_labels, test_data, test_labels, snp_ids = load_data_with_identifiers(
        args.data_file,
        datamodule_state,
    )
    
    # Compute baseline
    baseline = compute_baseline(train_data, baseline_type=args.baseline)
    
    # Compute Integrated Gradients
    attributions, deltas = compute_integrated_gradients(
        model,
        test_data,
        baseline,
        num_test=args.num_test,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        internal_batch_size=args.internal_batch_size,
    )
    
    # Get corresponding test labels for samples used
    torch.manual_seed(42)
    test_indices = torch.randperm(len(test_data))[:args.num_test]
    test_labels_used = test_labels[test_indices].numpy()
    
    # Aggregate and rank SNPs
    snp_importance_df = aggregate_and_rank_snps(
        attributions,
        deltas,
        snp_ids,
        output_path=str(output_dir / 'top_ig_snps.csv'),
    )
    
    # Save convergence deltas
    delta_df = pd.DataFrame({
        'Sample_Index': range(len(deltas)),
        'Convergence_Delta': deltas,
    })
    delta_df.to_csv(output_dir / 'convergence_deltas.csv', index=False)
    logger.info(f"✓ Convergence deltas saved to: {output_dir / 'convergence_deltas.csv'}")
    
    # Create visualizations
    create_bar_plot(
        snp_importance_df,
        output_path=str(output_dir / f'ig_bar_top{args.top_k_bar}.png'),
        top_k=args.top_k_bar,
    )
    
    create_heatmap(
        attributions,
        test_labels_used,
        snp_importance_df,
        output_path=str(output_dir / f'ig_heatmap_top{args.top_k_heatmap}.png'),
        top_k=args.top_k_heatmap,
        num_samples=30,
    )
    
    # Compare with SHAP if requested
    if args.compare_shap:
        shap_csv = Path(args.checkpoint_path).parent.parent / 'shap_analysis' / 'top_shap_snps.csv'
        compare_with_shap(
            snp_importance_df,
            str(shap_csv),
            output_path=str(output_dir / 'ig_vs_shap_comparison.png'),
            top_k=50,
        )
    
    logger.info("\n" + "="*80)
    logger.info("INTEGRATED GRADIENTS EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Ranked SNP list: top_ig_snps.csv")
    logger.info(f"  - Convergence deltas: convergence_deltas.csv")
    logger.info(f"  - Bar plot: ig_bar_top{args.top_k_bar}.png")
    logger.info(f"  - Heatmap: ig_heatmap_top{args.top_k_heatmap}.png")
    if args.compare_shap:
        logger.info(f"  - SHAP comparison: ig_vs_shap_comparison.png")
    logger.info("="*80)
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total SNPs analyzed: {len(snp_ids)}")
    logger.info(f"  Test samples explained: {len(attributions)}")
    logger.info(f"  Baseline type: {args.baseline}")
    logger.info(f"  Integration steps: {args.n_steps}")
    logger.info(f"  Mean importance (top 20): {snp_importance_df.head(20)['Mean_Abs_IG'].mean():.6f}")
    logger.info(f"  Mean importance (all): {snp_importance_df['Mean_Abs_IG'].mean():.6f}")
    logger.info(f"  Importance range: [{snp_importance_df['Mean_Abs_IG'].min():.6f}, {snp_importance_df['Mean_Abs_IG'].max():.6f}]")
    logger.info(f"  Convergence delta: mean={deltas.mean():.6f}, max={deltas.max():.6f}")


if __name__ == '__main__':
    main()
