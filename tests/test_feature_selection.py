"""Quick test script to verify feature selection implementation."""

import torch
import numpy as np
from src.data.feature_selection import (
    select_amgm,
    select_cosine_redundancy,
    select_variance_threshold,
    select_mutual_info,
    select_l1,
    select_features
)

def test_all_methods():
    """Test all feature selection methods with synthetic data."""
    print("=" * 60)
    print("Testing Feature Selection Methods")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Generate data with some informative and some noise features
    data = np.random.randn(n_samples, n_features)
    
    # Make first 10 features informative
    labels = np.random.randint(0, 2, n_samples)
    for i in range(10):
        data[labels == 1, i] += 2.0  # Shift for class 1
    
    print(f"\nSynthetic data: {n_samples} samples, {n_features} features")
    print(f"Labels: {np.sum(labels == 0)} class 0, {np.sum(labels == 1)} class 1")
    
    # Test AMGM
    print("\n" + "-" * 60)
    print("1. Testing AMGM (Arithmetic-Geometric Mean Ratio)")
    print("-" * 60)
    try:
        selected_idx, scores = select_amgm(data, labels, k=10, mode="ratio")
        print(f"✓ AMGM selected {len(selected_idx)} features")
        print(f"  Selected indices: {selected_idx[:5]}... (showing first 5)")
        print(f"  Top 5 scores: {scores[selected_idx[:5]]}")
    except Exception as e:
        print(f"✗ AMGM failed: {e}")
    
    # Test Cosine Redundancy
    print("\n" + "-" * 60)
    print("2. Testing Cosine Redundancy")
    print("-" * 60)
    try:
        selected_idx, scores = select_cosine_redundancy(data, k=10, threshold=0.95, method="greedy")
        print(f"✓ Cosine redundancy selected {len(selected_idx)} features")
        print(f"  Selected indices: {selected_idx[:5]}... (showing first 5)")
        print(f"  Mean similarities: {scores[selected_idx[:5]]}")
    except Exception as e:
        print(f"✗ Cosine redundancy failed: {e}")
    
    # Test Variance Threshold
    print("\n" + "-" * 60)
    print("3. Testing Variance Threshold")
    print("-" * 60)
    try:
        selected_idx, variances = select_variance_threshold(data, threshold=0.0, k=10)
        print(f"✓ Variance threshold selected {len(selected_idx)} features")
        print(f"  Selected indices: {selected_idx[:5]}... (showing first 5)")
        print(f"  Top 5 variances: {variances[selected_idx[:5]]}")
    except Exception as e:
        print(f"✗ Variance threshold failed: {e}")
    
    # Test Mutual Information
    print("\n" + "-" * 60)
    print("4. Testing Mutual Information")
    print("-" * 60)
    try:
        selected_idx, mi_scores = select_mutual_info(data, labels, k=10, random_state=42)
        print(f"✓ Mutual information selected {len(selected_idx)} features")
        print(f"  Selected indices: {selected_idx[:5]}... (showing first 5)")
        print(f"  Top 5 MI scores: {mi_scores[selected_idx[:5]]}")
    except Exception as e:
        print(f"✗ Mutual information failed: {e}")
    
    # Test L1 Regularization
    print("\n" + "-" * 60)
    print("5. Testing L1 Regularization")
    print("-" * 60)
    try:
        selected_idx, coeffs = select_l1(data, labels, k=10, C=1.0, random_state=42)
        print(f"✓ L1 regularization selected {len(selected_idx)} features")
        print(f"  Selected indices: {selected_idx[:5]}... (showing first 5)")
        print(f"  Top 5 coefficients: {coeffs[selected_idx[:5]]}")
    except Exception as e:
        print(f"✗ L1 regularization failed: {e}")
    
    # Test unified interface
    print("\n" + "-" * 60)
    print("6. Testing Unified Interface (select_features)")
    print("-" * 60)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    methods_to_test = [
        ("amgm", {"k": 10, "mode": "ratio"}),
        ("cosine", {"k": 10, "threshold": 0.95}),
        ("variance", {"k": 10}),
        ("mutual_info", {"k": 10}),
        ("l1", {"k": 10, "C": 1.0}),
    ]
    
    for method_name, params in methods_to_test:
        try:
            selected_data, selected_idx, scores = select_features(
                data_tensor, labels_tensor, method_name, **params
            )
            print(f"  ✓ {method_name:15s}: {selected_data.shape[1]} features selected")
        except Exception as e:
            print(f"  ✗ {method_name:15s}: {e}")
    
    print("\n" + "=" * 60)
    print("Feature Selection Tests Complete!")
    print("=" * 60)

def test_pipeline_selection():
    """Test multi-stage pipeline feature selection."""
    print("\n" + "=" * 60)
    print("Testing Pipeline (Multi-Stage) Feature Selection")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 1000
    
    # Generate data with some informative features
    data = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 2, n_samples)
    
    # Make first 50 features informative
    for i in range(50):
        data[labels == 1, i] += 2.0
    
    print(f"\nSynthetic data: {n_samples} samples, {n_features} features")
    print(f"Labels: {np.sum(labels == 0)} class 0, {np.sum(labels == 1)} class 1")
    
    # Test 3-stage pipeline: variance -> AMGM -> cosine
    print("\n" + "-" * 60)
    print("Testing 3-Stage Pipeline: Variance -> AMGM -> Cosine")
    print("-" * 60)
    
    stages = [
        {
            'name': 'variance_filter',
            'method': 'variance',
            'k': 500
        },
        {
            'name': 'amgm_selection',
            'method': 'amgm',
            'k': 100,
            'mode': 'ratio'
        },
        {
            'name': 'redundancy_removal',
            'method': 'cosine',
            'k': 20,
            'threshold': 0.95
        }
    ]
    
    try:
        from src.data.feature_selection import select_pipeline
        
        selected_indices, final_scores, stage_info = select_pipeline(data, labels, stages)
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Final feature count: {len(selected_indices)}")
        print(f"  Selected indices (first 10): {selected_indices[:10]}")
        
        print(f"\n  Stage Summary:")
        for i, info in enumerate(stage_info):
            print(f"    Stage {i+1} ({info['name']}): {info['features_in']} -> {info['features_out']} features")
        
        # Verify indices are in original space
        assert len(selected_indices) == 20, f"Expected 20 features, got {len(selected_indices)}"
        assert np.all(selected_indices < n_features), "Indices out of range"
        assert np.all(selected_indices >= 0), "Negative indices found"
        
        print(f"\n  ✓ Index validation passed")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test unified interface with pipeline
    print("\n" + "-" * 60)
    print("Testing Pipeline via Unified Interface")
    print("-" * 60)
    
    try:
        from src.data.feature_selection import select_features
        import torch
        
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        stage_info_out = []
        selected_data, selected_idx, scores = select_features(
            data_tensor, labels_tensor,
            method="pipeline",
            stages=stages,
            _stage_info_out=stage_info_out
        )
        
        print(f"✓ Unified interface pipeline: {selected_data.shape[1]} features selected")
        print(f"  Stages executed: {len(stage_info_out)}")
        
    except Exception as e:
        print(f"✗ Unified interface failed: {e}")
    
    print("\n" + "=" * 60)
    print("Pipeline Tests Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_all_methods()
    test_pipeline_selection()
