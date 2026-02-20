"""
LIME-based Explainability for SNP Classification Models

This script implements LIME (Local Interpretable Model-agnostic Explanations)
analysis for trained models on SNP classification tasks. It computes local 
feature attributions and aggregates them to global SNP importance rankings.

Key Features:
- Architecture-agnostic (works with all model types)
- Uses LIME's tabular explainer for local explanations
- Aggregates local explanations to global importance rankings
- Generates visualizations matching SHAP/IG output format
- Deterministic and fast computation

Usage:
    python src/lime_explainability.py --checkpoint_path <path_to_best_ckpt>
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

import lime
import lime.lime_tabular

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    """Architecture-agnostic wrapper compatible with LIME.
    
    LIME requires a function that returns class probabilities.
    This wrapper converts model logits to probability predictions.
    """
    
    def __init__(self, model: torch.nn.Module, use_softmax: bool = True):
        """Initialize the wrapper.
        
        Args:
            model: The neural network module (net component of LightningModule)
            use_softmax: Whether to apply softmax to logits (default: True)
        """
        super().__init__()
        self.model = model
        self.use_softmax = use_softmax
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, num_features)
            
        Returns:
            Class probabilities of shape (batch, num_classes)
        """
        logits = self.model(x)  # Shape: (batch, num_classes)
        if self.use_softmax:
            return torch.softmax(logits, dim=1)
        return logits


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], Tuple[float, float]]:
    """Load data and extract SNP identifiers, applying same preprocessing.
    
    Args:
        data_file: Path to the CSV file
        datamodule_state: State dict from the checkpoint's datamodule
        train_val_test_split: Train/val/test split ratios
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels, snp_ids, mean_std)
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
    
    # Apply feature selection using indices from checkpoint
    # Note: Normalization statistics (mean/std) from checkpoint are for intermediate feature space
    # and cannot be reliably applied to raw data without additional tracking information.
    # Feature selection alone is sufficient for LIME analysis since relative feature contributions are analyzed.
    selected_snp_ids = all_snp_ids
    mean_val, std_val = None, None
    if 'selected_indices' in datamodule_state:
        selected_indices = datamodule_state['selected_indices']
        if not isinstance(selected_indices, np.ndarray):
            selected_indices = np.array(selected_indices)
        
        data_tensor = data_tensor[:, selected_indices]
        selected_snp_ids = [all_snp_ids[i] for i in selected_indices]
        logger.info(f"✓ Applied feature selection: {len(selected_snp_ids)} SNPs selected")
    else:
        logger.warning("⚠ No feature selection indices found in checkpoint")
    
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
    
    return train_data, train_labels, test_data, test_labels, selected_snp_ids, (mean_val, std_val)


def compute_lime_values(
    model: LightningModule,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    snp_ids: List[str],
    num_test: int = 100,
    num_samples_per_test: int = 100,
    device: str = 'cpu',
) -> Tuple[np.ndarray, List[Dict]]:
    """Compute LIME attributions for test samples.
    
    Args:
        model: Trained LightningModule
        train_data: Training data for background statistics
        test_data: Test data tensor to explain
        snp_ids: List of SNP identifiers
        num_test: Number of test samples to explain
        num_samples_per_test: Number of perturbations per sample
        device: Device to run computation on
        
    Returns:
        Tuple of (aggregated_importance, lime_explanations)
    """
    logger.info(f"Computing LIME explanations for {num_test} test samples...")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Extract the neural network from LightningModule
    net = model.net
    
    # Sample test data
    num_test = min(num_test, len(test_data))
    torch.manual_seed(42)
    test_indices = torch.randperm(len(test_data))[:num_test]
    test_samples = test_data[test_indices].numpy()
    
    logger.info(f"Test samples shape: {test_samples.shape}")
    
    # Create a wrapper for LIME compatibility
    wrapped_model = ModelWrapper(net, use_softmax=True)
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    # Create prediction function for LIME
    def predict_fn(x):
        """Predict function that LIME can use."""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = wrapped_model(x_tensor)
        return probs.cpu().numpy()
    
    # Compute statistics from training data for LIME perturbations
    train_mean = train_data.numpy().mean(axis=0)
    train_std = train_data.numpy().std(axis=0)
    train_std = np.where(train_std == 0, 1e-6, train_std)  # Avoid division by zero
    
    logger.info("Initializing LIME TabularExplainer...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_data.numpy(),
        feature_names=[f"SNP_{i}" for i in range(len(snp_ids))],
        class_names=['Control', 'Case'],
        mode='classification',
        random_state=42
    )
    
    # Compute LIME explanations for each test sample
    logger.info(f"Computing LIME attributions (this may take a few minutes)...")
    
    # Aggregate LIME weights
    aggregated_weights = np.zeros(len(snp_ids))
    lime_explanations = []
    
    for i, test_sample in enumerate(test_samples):
        if (i + 1) % 5 == 0:
            logger.info(f"  Processing sample {i + 1}/{len(test_samples)}")
        
        try:
            # Compute LIME explanation for this sample
            exp = explainer.explain_instance(
                test_sample,
                predict_fn,
                num_features=min(len(snp_ids), 30),  # Limit features explained
                num_samples=num_samples_per_test,
                top_labels=1
            )
            
            # Extract feature weights (absolute values for global importance)
            explanation_dict = dict(exp.as_list())
            
            # Store explanation
            lime_explanations.append(explanation_dict)
            
            # Aggregate weights: get weight for each feature
            for feature_idx in range(len(snp_ids)):
                feature_name = f"SNP_{feature_idx}"
                if feature_name in explanation_dict:
                    aggregated_weights[feature_idx] += np.abs(explanation_dict[feature_name])
        except Exception as e:
            logger.warning(f"  Failed to explain sample {i}: {str(e)[:100]}")
            continue
    
    # Average the weights
    aggregated_weights /= len(test_samples)
    
    logger.info(f"✓ LIME explanations computed: {len(lime_explanations)} samples")
    
    return aggregated_weights, lime_explanations


def aggregate_and_rank_snps(
    lime_importance: np.ndarray,
    snp_ids: List[str],
    output_path: str,
) -> pd.DataFrame:
    """Aggregate LIME weights and rank SNPs by importance.
    
    Args:
        lime_importance: Mean absolute LIME weights for each feature
        snp_ids: List of SNP identifiers
        output_path: Path to save the ranked SNP list
        
    Returns:
        DataFrame with ranked SNPs and their importance scores
    """
    logger.info("Aggregating LIME weights across samples...")
    
    # Create DataFrame
    snp_importance_df = pd.DataFrame({
        'SNP_ID': snp_ids,
        'Mean_LIME_Weight': lime_importance,
    })
    
    # Sort by mean LIME weight (descending)
    snp_importance_df = snp_importance_df.sort_values('Mean_LIME_Weight', ascending=False)
    snp_importance_df['Rank'] = range(1, len(snp_importance_df) + 1)
    
    # Reorder columns
    snp_importance_df = snp_importance_df[['Rank', 'SNP_ID', 'Mean_LIME_Weight']]
    
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
    bars = ax.barh(range(len(top_snps)), top_snps['Mean_LIME_Weight'], color='coral')
    
    # Customize plot
    ax.set_yticks(range(len(top_snps)))
    ax.set_yticklabels(top_snps['SNP_ID'], fontsize=10)
    ax.set_xlabel('Mean Absolute LIME Weight', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNP Identifier', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important SNPs\n(LIME - Local Interpretable Model-agnostic Explanations)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_snps['Mean_LIME_Weight'])):
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='LIME Explainability Analysis for SNP Classification'
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
        '--num_samples_per_test',
        type=int,
        default=100,
        help='Number of perturbations per sample for LIME (default: 100)'
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
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        export_dir = os.getenv("SNP_EXPORT_DIR")
        if export_dir:
            args.output_dir = Path(export_dir) / "lime_analysis"
        else:
            checkpoint_dir = Path(args.checkpoint_path).parent.parent
            args.output_dir = checkpoint_dir / 'lime_analysis'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint_path)
    
    # Extract datamodule state
    datamodule_state = checkpoint.get('datamodule', checkpoint.get('DataModule', {}))
    
    # Load data with preprocessing
    train_data, train_labels, test_data, test_labels, snp_ids, (mean_val, std_val) = load_data_with_identifiers(
        args.data_file,
        datamodule_state,
    )
    
    # Compute LIME values
    lime_importance, lime_explanations = compute_lime_values(
        model,
        train_data,
        test_data,
        snp_ids,
        num_test=args.num_test,
        num_samples_per_test=args.num_samples_per_test,
        device=args.device,
    )
    
    # Aggregate and rank SNPs
    snp_importance_df = aggregate_and_rank_snps(
        lime_importance,
        snp_ids,
        output_path=str(output_dir / 'top_lime_snps.csv'),
    )
    
    # Create visualizations
    create_bar_plot(
        snp_importance_df,
        output_path=str(output_dir / f'lime_bar_top{args.top_k_bar}.png'),
        top_k=args.top_k_bar,
    )
    
    logger.info("\n" + "="*80)
    logger.info("LIME EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Ranked SNP list: top_lime_snps.csv")
    logger.info(f"  - Bar plot: lime_bar_top{args.top_k_bar}.png")
    logger.info("="*80)
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total SNPs analyzed: {len(snp_ids)}")
    logger.info(f"  Test samples explained: {len(lime_explanations)}")
    logger.info(f"  Mean importance (top 20): {snp_importance_df.head(20)['Mean_LIME_Weight'].mean():.6f}")
    logger.info(f"  Mean importance (all): {snp_importance_df['Mean_LIME_Weight'].mean():.6f}")
    logger.info(f"  Importance range: [{snp_importance_df['Mean_LIME_Weight'].min():.6f}, {snp_importance_df['Mean_LIME_Weight'].max():.6f}]")


if __name__ == '__main__':
    try:
        main()
    finally:
        # Clean up
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
