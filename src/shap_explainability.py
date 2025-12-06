"""
SHAP-based Explainability for BiLSTM SNP Classification Model

This script implements SHAP (SHapley Additive exPlanations) analysis for the trained
BiLSTM model on the Autism SNP dataset. It computes feature attribution scores,
ranks SNPs by importance, and generates publication-ready visualizations.

Key Features:
- Loads best checkpoint from ModelCheckpoint callback
- Reshapes input for BiLSTM sequence processing
- Uses GradientExplainer for attribution computation
- Generates global importance rankings across test samples
- Maps importance back to SNP identifiers
- Creates bar plots and heatmaps for interpretation

Usage:
    python src/shap_explainability.py --checkpoint_path <path_to_best_ckpt>
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

import shap

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BiLSTMShapWrapper(torch.nn.Module):
    """Wrapper for BiLSTM model to ensure SHAP-compatible forward pass.
    
    This wrapper handles the reshaping from (batch, num_features) to 
    (batch, seq_len, window_size) required by the BiLSTM model.
    """
    
    def __init__(self, model: torch.nn.Module, window_size: int):
        """Initialize the wrapper.
        
        Args:
            model: The BiLSTM model (net component of LightningModule)
            window_size: Window size used by BiLSTM for sequence creation
        """
        super().__init__()
        self.model = model
        self.window_size = window_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, num_features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # The BiLSTM model handles reshaping internally
        return self.model(x)


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


def compute_shap_values(
    model: LightningModule,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    num_background: int = 50,
    num_test: int = 100,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SHAP values using GradientExplainer.
    
    Args:
        model: Trained LightningModule
        train_data: Training data tensor for background
        test_data: Test data tensor to explain
        num_background: Number of background samples
        num_test: Number of test samples to explain
        device: Device to run computation on
        
    Returns:
        Tuple of (shap_values, background_samples, test_samples)
    """
    logger.info("Computing SHAP values using GradientExplainer...")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Extract the neural network from LightningModule
    net = model.net
    
    # Sample background and test data
    num_background = min(num_background, len(train_data))
    num_test = min(num_test, len(test_data))
    
    # Random sampling
    torch.manual_seed(42)
    background_indices = torch.randperm(len(train_data))[:num_background]
    test_indices = torch.randperm(len(test_data))[:num_test]
    
    background_samples = train_data[background_indices].to(device)
    test_samples = test_data[test_indices].to(device)
    
    logger.info(f"Background samples: {background_samples.shape}")
    logger.info(f"Test samples: {test_samples.shape}")
    
    # Ensure gradients are enabled for inputs
    background_samples.requires_grad = True
    test_samples.requires_grad = True
    
    # Create wrapper for the network
    window_size = net.window_size if hasattr(net, 'window_size') else 50
    wrapped_model = BiLSTMShapWrapper(net, window_size)
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    # Initialize GradientExplainer
    logger.info("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(wrapped_model, background_samples)
    
    # Compute SHAP values
    logger.info("Computing SHAP attributions (this may take a few minutes)...")
    with torch.enable_grad():
        shap_values = explainer.shap_values(test_samples)
    
    # Convert to numpy
    if isinstance(shap_values, list):
        # Multi-class: take values for class 1 (case)
        shap_values_np = shap_values[1]
    else:
        shap_values_np = shap_values
    
    # Move to CPU and convert to numpy
    if isinstance(shap_values_np, torch.Tensor):
        shap_values_np = shap_values_np.cpu().detach().numpy()
    
    background_np = background_samples.cpu().detach().numpy()
    test_np = test_samples.cpu().detach().numpy()
    
    logger.info(f"✓ SHAP values computed: {shap_values_np.shape}")
    
    return shap_values_np, background_np, test_np


def aggregate_and_rank_snps(
    shap_values: np.ndarray,
    snp_ids: List[str],
    output_path: str,
) -> pd.DataFrame:
    """Aggregate SHAP values and rank SNPs by importance.
    
    Args:
        shap_values: SHAP values of shape (num_samples, num_features) or (num_samples, num_features, num_classes)
        snp_ids: List of SNP identifiers
        output_path: Path to save the ranked SNP list
        
    Returns:
        DataFrame with ranked SNPs and their importance scores
    """
    logger.info("Aggregating SHAP values across samples...")
    
    # Handle multi-class SHAP values - take class 1 (case) for binary classification
    if shap_values.ndim == 3:
        logger.info(f"Multi-class SHAP values detected (shape: {shap_values.shape}), using class 1 (case)")
        shap_values = shap_values[:, :, 1]  # Shape: (num_samples, num_features)
    
    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create DataFrame
    snp_importance_df = pd.DataFrame({
        'SNP_ID': snp_ids,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Mean_SHAP': np.mean(shap_values, axis=0),
        'Std_SHAP': np.std(shap_values, axis=0),
        'Max_Abs_SHAP': np.max(np.abs(shap_values), axis=0),
    })
    
    # Sort by mean absolute SHAP value (descending)
    snp_importance_df = snp_importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
    snp_importance_df['Rank'] = range(1, len(snp_importance_df) + 1)
    
    # Reorder columns
    snp_importance_df = snp_importance_df[['Rank', 'SNP_ID', 'Mean_Abs_SHAP', 'Mean_SHAP', 'Std_SHAP', 'Max_Abs_SHAP']]
    
    # Save to CSV
    snp_importance_df.to_csv(output_path, index=False)
    logger.info(f"✓ Ranked SNPs saved to: {output_path}")
    
    # Print top 20
    logger.info("\nTop 20 most important SNPs:")
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
    bars = ax.barh(range(len(top_snps)), top_snps['Mean_Abs_SHAP'], color='steelblue')
    
    # Customize plot
    ax.set_yticks(range(len(top_snps)))
    ax.set_yticklabels(top_snps['SNP_ID'], fontsize=10)
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNP Identifier', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important SNPs for Autism Classification\n(BiLSTM Model)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_snps['Mean_Abs_SHAP'])):
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
    shap_values: np.ndarray,
    test_labels: np.ndarray,
    snp_importance_df: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
    num_samples: int = 30,
):
    """Create heatmap showing per-sample SHAP contributions for top SNPs.
    
    Args:
        shap_values: SHAP values of shape (num_samples, num_features) or (num_samples, num_features, num_classes)
        test_labels: Test labels (0=control, 1=case)
        snp_importance_df: DataFrame with ranked SNPs
        output_path: Path to save the figure
        top_k: Number of top SNPs to display
        num_samples: Number of samples to display
    """
    logger.info(f"Creating heatmap for top {top_k} SNPs across {num_samples} samples...")
    
    # Handle multi-class SHAP values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # Use class 1 (case)
    
    # Get indices of top K SNPs
    top_snp_ids = snp_importance_df.head(top_k)['SNP_ID'].tolist()
    all_snp_ids = snp_importance_df['SNP_ID'].tolist()
    top_indices = [all_snp_ids.index(snp_id) for snp_id in top_snp_ids]
    
    # Select SHAP values for top SNPs
    shap_top = shap_values[:, top_indices]
    
    # Limit number of samples for visualization
    if shap_top.shape[0] > num_samples:
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
        
        shap_top = shap_top[selected_indices]
        test_labels_subset = test_labels[selected_indices]
    else:
        test_labels_subset = test_labels
    
    # Transpose for better visualization (SNPs as rows, samples as columns)
    shap_top = shap_top.T
    
    # Create sample labels
    sample_labels = [f"Case {i+1}" if label == 1 else f"Control {i+1}" 
                    for i, label in enumerate(test_labels_subset)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create heatmap
    sns.heatmap(
        shap_top,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'SHAP Value', 'shrink': 0.8},
        xticklabels=sample_labels,
        yticklabels=top_snp_ids if top_k <= 50 else False,
        ax=ax,
        vmin=-np.abs(shap_top).max(),
        vmax=np.abs(shap_top).max(),
    )
    
    # Customize plot
    ax.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNP Identifier', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Sample SHAP Contributions for Top {top_k} SNPs\n'
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='SHAP Explainability Analysis for BiLSTM SNP Classification'
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
        '--num_background',
        type=int,
        default=50,
        help='Number of background samples for SHAP (default: 50)'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=100,
        help='Number of test samples to explain (default: 100)'
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
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint_path).parent.parent
        args.output_dir = checkpoint_dir / 'shap_analysis'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint_path)
    
    # Extract datamodule state - check both possible keys
    datamodule_state = checkpoint.get('datamodule', checkpoint.get('DataModule', {}))
    
    # Load data with preprocessing
    train_data, train_labels, test_data, test_labels, snp_ids = load_data_with_identifiers(
        args.data_file,
        datamodule_state,
    )
    
    # Compute SHAP values
    shap_values, background_samples, test_samples = compute_shap_values(
        model,
        train_data,
        test_data,
        num_background=args.num_background,
        num_test=args.num_test,
        device=args.device,
    )
    
    # Get corresponding test labels for samples used
    torch.manual_seed(42)
    test_indices = torch.randperm(len(test_data))[:args.num_test]
    test_labels_used = test_labels[test_indices].numpy()
    
    # Aggregate and rank SNPs
    snp_importance_df = aggregate_and_rank_snps(
        shap_values,
        snp_ids,
        output_path=str(output_dir / 'top_shap_snps.csv'),
    )
    
    # Create visualizations
    create_bar_plot(
        snp_importance_df,
        output_path=str(output_dir / f'shap_bar_top{args.top_k_bar}.png'),
        top_k=args.top_k_bar,
    )
    
    create_heatmap(
        shap_values,
        test_labels_used,
        snp_importance_df,
        output_path=str(output_dir / f'shap_heatmap_top{args.top_k_heatmap}.png'),
        top_k=args.top_k_heatmap,
        num_samples=30,
    )
    
    logger.info("\n" + "="*80)
    logger.info("SHAP EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Ranked SNP list: top_shap_snps.csv")
    logger.info(f"  - Bar plot: shap_bar_top{args.top_k_bar}.png")
    logger.info(f"  - Heatmap: shap_heatmap_top{args.top_k_heatmap}.png")
    logger.info("="*80)
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total SNPs analyzed: {len(snp_ids)}")
    logger.info(f"  Test samples explained: {len(shap_values)}")
    logger.info(f"  Mean importance (top 20): {snp_importance_df.head(20)['Mean_Abs_SHAP'].mean():.6f}")
    logger.info(f"  Mean importance (all): {snp_importance_df['Mean_Abs_SHAP'].mean():.6f}")
    logger.info(f"  Importance range: [{snp_importance_df['Mean_Abs_SHAP'].min():.6f}, {snp_importance_df['Mean_Abs_SHAP'].max():.6f}]")


if __name__ == '__main__':
    main()
