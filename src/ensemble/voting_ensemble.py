"""Weighted soft-voting ensemble for SNP classification models.

This module implements a weighted soft-voting ensemble that combines predictions
from multiple pre-trained models using validation accuracy as weights.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

from src.models.module import LitModule

log = logging.getLogger(__name__)


class WeightedSoftVotingEnsemble:
    """Weighted soft-voting ensemble for deep learning models.
    
    This ensemble loads multiple pre-trained PyTorch Lightning models and combines
    their predictions using weighted soft voting, where each model's weight is
    proportional to its validation accuracy.
    
    Attributes:
        models: List of loaded PyTorch Lightning models.
        weights: Normalized weights for each model (sum to 1.0).
        model_names: Names of the models in the ensemble.
        num_classes: Number of output classes.
        device: Device to run inference on.
    """
    
    def __init__(
        self,
        checkpoint_paths: Dict[str, str],
        validation_accuracies: Dict[str, float],
        device: str = "cpu",
        num_classes: int = 2,
    ):
        """Initialize the weighted soft-voting ensemble.
        
        Args:
            checkpoint_paths: Dictionary mapping model names to checkpoint paths.
            validation_accuracies: Dictionary mapping model names to validation accuracies.
            device: Device to run inference on ('cpu' or 'cuda').
            num_classes: Number of output classes.
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model_names = list(checkpoint_paths.keys())
        
        # Validate inputs
        if set(checkpoint_paths.keys()) != set(validation_accuracies.keys()):
            raise ValueError(
                "Model names in checkpoint_paths and validation_accuracies must match. "
                f"Got {set(checkpoint_paths.keys())} vs {set(validation_accuracies.keys())}"
            )
        
        if not checkpoint_paths:
            raise ValueError("At least one model checkpoint must be provided.")
        
        # Load models
        log.info(f"Loading {len(checkpoint_paths)} models for ensemble...")
        self.models = self._load_models(checkpoint_paths)
        
        # Compute normalized weights
        self.weights = self._compute_weights(validation_accuracies)
        
        log.info("Ensemble initialized successfully.")
        log.info(f"Model weights: {dict(zip(self.model_names, self.weights))}")
    
    def _load_models(self, checkpoint_paths: Dict[str, str]) -> List[LitModule]:
        """Load all models from checkpoints.
        
        Args:
            checkpoint_paths: Dictionary mapping model names to checkpoint paths.
            
        Returns:
            List of loaded PyTorch Lightning models.
        """
        models = []
        
        for model_name, ckpt_path in checkpoint_paths.items():
            # Verify checkpoint exists
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
            log.info(f"Loading {model_name} from {ckpt_path}...")
            
            try:
                # Load model from checkpoint
                model = LitModule.load_from_checkpoint(
                    ckpt_path,
                    map_location=self.device,
                )
                model.eval()  # Set to evaluation mode
                model.to(self.device)
                
                models.append(model)
                log.info(f"  ✓ {model_name} loaded successfully.")
                
            except Exception as e:
                log.error(f"  ✗ Failed to load {model_name}: {str(e)}")
                raise
        
        return models
    
    def _compute_weights(self, validation_accuracies: Dict[str, float]) -> np.ndarray:
        """Compute normalized weights based on validation accuracies.
        
        Uses the formula: w_i = Acc_i / sum(Acc_j)
        
        Args:
            validation_accuracies: Dictionary mapping model names to validation accuracies.
            
        Returns:
            Numpy array of normalized weights (sum to 1.0).
        """
        # Extract accuracies in the same order as model_names
        accuracies = np.array([validation_accuracies[name] for name in self.model_names])
        
        # Validate accuracies
        if np.any(accuracies <= 0):
            raise ValueError("All validation accuracies must be positive.")
        
        if np.any(accuracies > 1.0):
            log.warning(
                "Validation accuracies > 1.0 detected. "
                "Assuming they are percentages and dividing by 100."
            )
            accuracies = accuracies / 100.0
        
        # Compute normalized weights
        weights = accuracies / accuracies.sum()
        
        return weights
    
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute weighted ensemble probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, n_features).
            
        Returns:
            Tuple of:
                - Weighted ensemble probabilities of shape (batch_size, num_classes).
                - List of individual model probabilities (one per model).
        """
        x = x.to(self.device)
        batch_size = x.size(0)
        
        # Initialize ensemble probabilities
        ensemble_probs = torch.zeros(
            batch_size, self.num_classes,
            device=self.device,
            dtype=torch.float32
        )
        
        # Collect individual model probabilities
        individual_probs = []
        
        # Aggregate predictions from all models
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            # Get logits from model
            logits = model(x)
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Add weighted probabilities to ensemble
            ensemble_probs += weight * probs
            
            # Store individual probabilities
            individual_probs.append(probs.cpu())
        
        return ensemble_probs.cpu(), individual_probs
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Compute weighted ensemble predictions.
        
        Args:
            x: Input tensor of shape (batch_size, n_features).
            
        Returns:
            Tuple of:
                - Ensemble predictions of shape (batch_size,).
                - Ensemble probabilities of shape (batch_size, num_classes).
                - List of individual model probabilities (one per model).
        """
        # Get ensemble probabilities
        ensemble_probs, individual_probs = self.predict_proba(x)
        
        # Get final predictions via argmax
        ensemble_preds = torch.argmax(ensemble_probs, dim=1)
        
        return ensemble_preds, ensemble_probs, individual_probs
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the ensemble models.
        
        Returns:
            Dictionary containing model names, weights, and other metadata.
        """
        return {
            "model_names": self.model_names,
            "weights": self.weights.tolist(),
            "num_models": len(self.models),
            "num_classes": self.num_classes,
            "device": str(self.device),
        }
    
    def __repr__(self) -> str:
        """String representation of the ensemble."""
        model_info = "\n".join([
            f"  - {name}: weight={weight:.4f}"
            for name, weight in zip(self.model_names, self.weights)
        ])
        return (
            f"WeightedSoftVotingEnsemble(\n"
            f"  num_models={len(self.models)},\n"
            f"  num_classes={self.num_classes},\n"
            f"  device={self.device},\n"
            f"  models=[\n{model_info}\n  ]\n"
            f")"
        )


def create_ensemble_from_config(
    model_configs: List[Dict[str, any]],
    weighting_strategy: str = "accuracy",
    device: str = "cpu",
    num_classes: int = 2,
) -> WeightedSoftVotingEnsemble:
    """Create ensemble from a list of model configurations.
    
    Args:
        model_configs: List of dictionaries, each containing:
            - name: Model name
            - checkpoint_path: Path to checkpoint
            - val_accuracy: Validation accuracy (if weighting_strategy="accuracy")
            - weight: Custom weight (if weighting_strategy="custom")
        weighting_strategy: Strategy for computing weights ("accuracy" or "custom").
        device: Device to run inference on.
        num_classes: Number of output classes.
        
    Returns:
        Initialized WeightedSoftVotingEnsemble instance.
    """
    checkpoint_paths = {}
    validation_accuracies = {}
    
    for config in model_configs:
        name = config["name"]
        checkpoint_paths[name] = config["checkpoint_path"]
        
        if weighting_strategy == "accuracy":
            validation_accuracies[name] = config["val_accuracy"]
        elif weighting_strategy == "custom":
            validation_accuracies[name] = config.get("weight", 1.0)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {weighting_strategy}. "
                "Must be 'accuracy' or 'custom'."
            )
    
    return WeightedSoftVotingEnsemble(
        checkpoint_paths=checkpoint_paths,
        validation_accuracies=validation_accuracies,
        device=device,
        num_classes=num_classes,
    )
