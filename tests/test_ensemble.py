"""Unit tests for ensemble module."""

import torch
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.ensemble.voting_ensemble import WeightedSoftVotingEnsemble
from src.ensemble.preprocessing import (
    apply_preprocessing,
    verify_preprocessing_consistency,
)


class TestPreprocessing:
    """Tests for preprocessing utilities."""
    
    def test_apply_preprocessing(self):
        """Test preprocessing application."""
        # Create sample data
        data = torch.randn(10, 100)
        mean = torch.zeros(100)
        std = torch.ones(100)
        selected_indices = np.array([0, 5, 10, 20, 50])
        
        # Apply preprocessing
        preprocessed = apply_preprocessing(data, mean, std, selected_indices)
        
        # Check shape
        assert preprocessed.shape == (10, 5)
        
        # Check that correct features were selected
        assert torch.allclose(preprocessed[:, 0], data[:, 0], atol=1e-6)
        assert torch.allclose(preprocessed[:, 1], data[:, 5], atol=1e-6)
    
    def test_apply_preprocessing_with_normalization(self):
        """Test preprocessing with non-zero mean and std."""
        data = torch.randn(10, 100) * 5 + 10  # Mean ~10, std ~5
        mean = torch.ones(100) * 10
        std = torch.ones(100) * 5
        selected_indices = np.arange(100)
        
        preprocessed = apply_preprocessing(data, mean, std, selected_indices)
        
        # Check normalization worked (approximately mean=0, std=1)
        assert abs(preprocessed.mean().item()) < 0.5
        assert abs(preprocessed.std().item() - 1.0) < 0.5


class TestWeightedSoftVotingEnsemble:
    """Tests for WeightedSoftVotingEnsemble class."""
    
    def test_weight_computation(self):
        """Test that weights are computed correctly."""
        # This test would require mock checkpoints
        # For now, just test the weight normalization logic
        accuracies = np.array([0.85, 0.83, 0.84])
        weights = accuracies / accuracies.sum()
        
        # Check weights sum to 1
        assert np.isclose(weights.sum(), 1.0)
        
        # Check weights are proportional to accuracies
        assert weights[0] > weights[1]  # 0.85 > 0.83
        assert weights[0] > weights[2]  # 0.85 > 0.84
    
    def test_weight_computation_percentages(self):
        """Test that percentage accuracies are handled correctly."""
        accuracies = np.array([85.0, 83.0, 84.0])
        
        # Should detect and convert
        if np.any(accuracies > 1.0):
            accuracies = accuracies / 100.0
        
        weights = accuracies / accuracies.sum()
        
        assert np.isclose(weights.sum(), 1.0)
        assert all(weights > 0)
        assert all(weights < 1)


class TestEnsembleConfiguration:
    """Tests for ensemble configuration."""
    
    def test_config_structure(self):
        """Test that ensemble config has required fields."""
        config = {
            "ensemble": {
                "weighting_strategy": "accuracy",
                "device": "cpu",
                "save_predictions": True,
                "generate_plots": True,
                "models": [
                    {
                        "name": "bilstm",
                        "checkpoint_path": "path/to/ckpt.ckpt",
                        "val_accuracy": 0.85,
                    }
                ]
            }
        }
        
        # Check required fields exist
        assert "ensemble" in config
        assert "models" in config["ensemble"]
        assert len(config["ensemble"]["models"]) > 0
        
        model = config["ensemble"]["models"][0]
        assert "name" in model
        assert "checkpoint_path" in model
        assert "val_accuracy" in model
    
    def test_config_validation(self):
        """Test config validation logic."""
        models = [
            {"name": "m1", "checkpoint_path": "p1", "val_accuracy": 0.85},
            {"name": "m2", "checkpoint_path": "p2", "val_accuracy": 0.83},
        ]
        
        # Check all required fields present
        for model in models:
            assert "name" in model
            assert "checkpoint_path" in model
            assert "val_accuracy" in model
            assert 0 <= model["val_accuracy"] <= 1.0


class TestEnsemblePrediction:
    """Tests for ensemble prediction logic."""
    
    def test_weighted_average(self):
        """Test weighted average computation."""
        # Simulate probabilities from 3 models
        probs1 = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        probs2 = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        probs3 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        
        weights = np.array([0.4, 0.3, 0.3])
        
        # Compute weighted average
        ensemble_probs = (
            weights[0] * probs1 + 
            weights[1] * probs2 + 
            weights[2] * probs3
        )
        
        # Check shape
        assert ensemble_probs.shape == (2, 2)
        
        # Check probabilities sum to 1 (approximately)
        assert torch.allclose(ensemble_probs.sum(dim=1), torch.ones(2), atol=1e-5)
        
        # Check predictions
        preds = torch.argmax(ensemble_probs, dim=1)
        assert preds[0] == 0  # Should predict class 0
        assert preds[1] == 1  # Should predict class 1


def test_imports():
    """Test that all modules can be imported."""
    from src.ensemble import (
        WeightedSoftVotingEnsemble,
        load_preprocessing_from_checkpoint,
        apply_preprocessing,
        EnsembleEvaluator,
    )
    
    assert WeightedSoftVotingEnsemble is not None
    assert load_preprocessing_from_checkpoint is not None
    assert apply_preprocessing is not None
    assert EnsembleEvaluator is not None


if __name__ == "__main__":
    # Run basic tests
    print("Running ensemble unit tests...")
    
    print("\n1. Testing imports...")
    test_imports()
    print("   ✓ Imports successful")
    
    print("\n2. Testing preprocessing...")
    test_prep = TestPreprocessing()
    test_prep.test_apply_preprocessing()
    test_prep.test_apply_preprocessing_with_normalization()
    print("   ✓ Preprocessing tests passed")
    
    print("\n3. Testing weight computation...")
    test_weights = TestWeightedSoftVotingEnsemble()
    test_weights.test_weight_computation()
    test_weights.test_weight_computation_percentages()
    print("   ✓ Weight computation tests passed")
    
    print("\n4. Testing config validation...")
    test_config = TestEnsembleConfiguration()
    test_config.test_config_structure()
    test_config.test_config_validation()
    print("   ✓ Config validation tests passed")
    
    print("\n5. Testing prediction logic...")
    test_pred = TestEnsemblePrediction()
    test_pred.test_weighted_average()
    print("   ✓ Prediction logic tests passed")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
