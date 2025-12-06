"""Preprocessing utilities for ensemble models.

This module ensures that all ensemble models use identical preprocessing
(normalization and feature selection) as applied during training.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
from pathlib import Path


def load_preprocessing_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load preprocessing parameters from a model checkpoint.
    
    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint file.
        
    Returns:
        Dictionary containing:
            - mean: Normalization mean (torch.Tensor)
            - std: Normalization std (torch.Tensor)
            - selected_indices: Feature selection indices (np.ndarray)
            - feature_importances: Feature importance scores (np.ndarray, optional)
            - pipeline_info: Feature selection pipeline information (dict, optional)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try both 'datamodule' and 'DataModule' keys
    datamodule_state = checkpoint.get('datamodule', checkpoint.get('DataModule', {}))
    
    if not datamodule_state:
        raise ValueError(
            f"No datamodule state found in checkpoint {checkpoint_path}. "
            "Ensure the checkpoint was saved with datamodule state."
        )
    
    preprocessing_params = {}
    
    # Extract normalization parameters
    mean = datamodule_state.get('mean')
    std = datamodule_state.get('std')
    
    if mean is None or std is None:
        raise ValueError(
            f"Normalization parameters (mean, std) not found in checkpoint {checkpoint_path}."
        )
    
    preprocessing_params['mean'] = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean)
    preprocessing_params['std'] = std if isinstance(std, torch.Tensor) else torch.tensor(std)
    
    # Extract feature selection indices
    selected_indices = datamodule_state.get('selected_indices')
    if selected_indices is None:
        raise ValueError(
            f"Feature selection indices not found in checkpoint {checkpoint_path}."
        )
    
    preprocessing_params['selected_indices'] = (
        selected_indices if isinstance(selected_indices, np.ndarray) 
        else np.array(selected_indices)
    )
    
    # Optional: Extract feature importances and pipeline info
    feature_importances = datamodule_state.get('feature_importances')
    if feature_importances is not None:
        preprocessing_params['feature_importances'] = (
            feature_importances if isinstance(feature_importances, np.ndarray)
            else np.array(feature_importances)
        )
    
    pipeline_info = datamodule_state.get('pipeline_info')
    if pipeline_info is not None:
        preprocessing_params['pipeline_info'] = pipeline_info
    
    return preprocessing_params


def apply_preprocessing(
    data: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    selected_indices: np.ndarray,
) -> torch.Tensor:
    """Apply normalization and feature selection to data.
    
    Args:
        data: Input data tensor of shape (n_samples, n_features).
        mean: Normalization mean of shape (1, n_features) or (n_features,).
        std: Normalization std of shape (1, n_features) or (n_features,).
        selected_indices: Indices of selected features (length k).
        
    Returns:
        Preprocessed data tensor of shape (n_samples, k).
    """
    # Ensure mean and std have correct shape
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    if std.dim() == 1:
        std = std.unsqueeze(0)
    
    # Move to same device as data
    mean = mean.to(data.device)
    std = std.to(data.device)
    
    # Apply Z-score normalization
    normalized_data = (data - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
    
    # Apply feature selection
    selected_data = normalized_data[:, selected_indices]
    
    return selected_data


def verify_preprocessing_consistency(checkpoint_paths: list[str]) -> Tuple[bool, str]:
    """Verify that all checkpoints use the same preprocessing parameters.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files.
        
    Returns:
        Tuple of (is_consistent, message).
    """
    if not checkpoint_paths:
        return False, "No checkpoint paths provided."
    
    try:
        # Load preprocessing from first checkpoint as reference
        reference_params = load_preprocessing_from_checkpoint(checkpoint_paths[0])
        reference_indices = reference_params['selected_indices']
        reference_mean = reference_params['mean']
        reference_std = reference_params['std']
        
        # Check consistency with other checkpoints
        for i, ckpt_path in enumerate(checkpoint_paths[1:], start=1):
            current_params = load_preprocessing_from_checkpoint(ckpt_path)
            
            # Check feature selection indices
            if not np.array_equal(reference_indices, current_params['selected_indices']):
                return False, (
                    f"Feature selection indices mismatch between checkpoint 0 and {i}. "
                    f"Reference has {len(reference_indices)} features, "
                    f"checkpoint {i} has {len(current_params['selected_indices'])} features."
                )
            
            # Check normalization parameters (allow small numerical differences)
            mean_diff = torch.abs(reference_mean - current_params['mean']).max().item()
            std_diff = torch.abs(reference_std - current_params['std']).max().item()
            
            if mean_diff > 1e-5 or std_diff > 1e-5:
                return False, (
                    f"Normalization parameters mismatch between checkpoint 0 and {i}. "
                    f"Max mean diff: {mean_diff:.2e}, Max std diff: {std_diff:.2e}"
                )
        
        return True, f"All {len(checkpoint_paths)} checkpoints use consistent preprocessing."
        
    except Exception as e:
        return False, f"Error verifying preprocessing consistency: {str(e)}"


def get_preprocessing_summary(preprocessing_params: Dict[str, Any]) -> str:
    """Generate a human-readable summary of preprocessing parameters.
    
    Args:
        preprocessing_params: Dictionary of preprocessing parameters.
        
    Returns:
        Formatted summary string.
    """
    lines = ["Preprocessing Configuration:"]
    lines.append("-" * 50)
    
    # Normalization info
    mean = preprocessing_params['mean']
    std = preprocessing_params['std']
    lines.append(f"Normalization: Z-score (mean=0, std=1)")
    lines.append(f"  - Original features: {len(mean.squeeze())}")
    lines.append(f"  - Mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    lines.append(f"  - Std range: [{std.min().item():.4f}, {std.max().item():.4f}]")
    
    # Feature selection info
    selected_indices = preprocessing_params['selected_indices']
    lines.append(f"\nFeature Selection:")
    lines.append(f"  - Selected features: {len(selected_indices)}")
    lines.append(f"  - Index range: [{selected_indices.min()}, {selected_indices.max()}]")
    
    # Pipeline info if available
    if 'pipeline_info' in preprocessing_params:
        pipeline_info = preprocessing_params['pipeline_info']
        lines.append(f"\nFeature Selection Pipeline:")
        for stage in pipeline_info.get('stages', []):
            lines.append(f"  - {stage.get('name', 'unknown')}: {stage.get('method', 'unknown')}")
    
    return "\n".join(lines)
