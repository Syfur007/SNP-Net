"""Feature selection methods for SNP data.

This module implements various feature selection strategies:
- AMGM (Arithmetic-Geometric Mean Ratio): Filter method based on class-separated means
- Cosine Redundancy: Removes highly correlated features
- Variance Threshold: Removes low-variance features
- Mutual Information: Ranks features by MI with target
- L1 Regularization: Embedded method using Lasso/LogisticRegression
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold as SKVarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def select_amgm(
    data: np.ndarray,
    labels: np.ndarray,
    k: int,
    mode: str = "ratio"
) -> Tuple[np.ndarray, np.ndarray]:
    """Select features using Arithmetic-Geometric Mean (AMGM) ratio.
    
    Computes the ratio of arithmetic mean to geometric mean for each class,
    then ranks features by the difference between class ratios.
    
    :param data: Feature matrix of shape (n_samples, n_features).
    :param labels: Labels of shape (n_samples,).
    :param k: Number of features to select.
    :param mode: Selection mode - "ratio" (AM/GM ratio) or "diff" (class mean difference).
    :return: Tuple of (selected_indices, scores) where scores are the AMGM values.
    """
    n_samples, n_features = data.shape
    unique_labels = np.unique(labels)
    
    if len(unique_labels) != 2:
        raise ValueError(f"AMGM currently supports binary classification only. Found {len(unique_labels)} classes.")
    
    scores = np.zeros(n_features)
    
    for i in range(n_features):
        feature = data[:, i]
        
        # Separate by class
        class0_vals = feature[labels == unique_labels[0]]
        class1_vals = feature[labels == unique_labels[1]]
        
        # Add small epsilon to avoid log(0) in geometric mean
        epsilon = 1e-10
        class0_vals = np.abs(class0_vals) + epsilon
        class1_vals = np.abs(class1_vals) + epsilon
        
        if mode == "ratio":
            # Arithmetic mean / Geometric mean ratio
            am0 = np.mean(class0_vals)
            gm0 = np.exp(np.mean(np.log(class0_vals)))
            ratio0 = am0 / gm0
            
            am1 = np.mean(class1_vals)
            gm1 = np.exp(np.mean(np.log(class1_vals)))
            ratio1 = am1 / gm1
            
            # Score is the absolute difference between ratios
            scores[i] = np.abs(ratio0 - ratio1)
        elif mode == "diff":
            # Simple mean difference between classes
            scores[i] = np.abs(np.mean(class0_vals) - np.mean(class1_vals))
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'ratio' or 'diff'.")
    
    # Select top-k features
    k = min(k, n_features)
    selected_indices = np.argsort(scores)[-k:][::-1]  # Descending order
    selected_indices = np.sort(selected_indices)  # Sort for contiguous indexing
    
    return selected_indices, scores


def select_cosine_redundancy(
    data: np.ndarray,
    k: int,
    threshold: float = 0.95,
    method: str = "greedy"
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove redundant features based on cosine similarity.
    
    Computes pairwise cosine similarity between features and removes
    highly correlated ones, keeping the most representative features.
    
    :param data: Feature matrix of shape (n_samples, n_features).
    :param k: Target number of features to select.
    :param threshold: Cosine similarity threshold above which features are considered redundant.
    :param method: Selection method - "greedy" (iterative removal) or "cluster" (clustering-based).
    :return: Tuple of (selected_indices, scores) where scores are mean cosine similarities.
    """
    n_samples, n_features = data.shape
    
    # Normalize features for cosine similarity
    data_normalized = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-10)
    
    # Compute pairwise cosine similarity matrix
    similarity_matrix = np.abs(data_normalized.T @ data_normalized)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    
    if method == "greedy":
        # Greedy approach: iteratively remove features with highest average similarity
        remaining_indices = np.arange(n_features)
        
        while len(remaining_indices) > k:
            # Compute mean similarity for each remaining feature
            mean_similarities = similarity_matrix[remaining_indices][:, remaining_indices].mean(axis=1)
            
            # Remove feature with highest mean similarity
            worst_idx_in_remaining = np.argmax(mean_similarities)
            worst_idx = remaining_indices[worst_idx_in_remaining]
            
            remaining_indices = np.delete(remaining_indices, worst_idx_in_remaining)
        
        selected_indices = np.sort(remaining_indices)  # Sort for contiguous indexing
        scores = similarity_matrix.mean(axis=1)
        
    elif method == "cluster":
        # Cluster-based: identify redundant groups and keep one representative per group
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Cluster features
        n_clusters = min(k, n_features)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Select one representative per cluster (lowest mean distance to others in cluster)
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 1:
                selected_indices.append(cluster_indices[0])
            else:
                # Select feature with lowest mean distance to other cluster members
                within_cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)].mean(axis=1)
                best_in_cluster = cluster_indices[np.argmin(within_cluster_distances)]
                selected_indices.append(best_in_cluster)
        
        selected_indices = np.sort(np.array(selected_indices))  # Sort for contiguous indexing
        scores = similarity_matrix.mean(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'cluster'.")
    
    return selected_indices, scores


def select_variance_threshold(
    data: np.ndarray,
    threshold: float = 0.0,
    k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Select features based on variance threshold.
    
    Removes features with variance below the threshold. If k is specified,
    selects top-k features by variance.
    
    :param data: Feature matrix of shape (n_samples, n_features).
    :param threshold: Minimum variance threshold.
    :param k: Optional number of features to select (top-k by variance).
    :return: Tuple of (selected_indices, variances).
    """
    n_samples, n_features = data.shape
    
    # Compute variances
    variances = np.var(data, axis=0)
    
    if k is not None:
        # Select top-k by variance
        k = min(k, n_features)
        selected_indices = np.argsort(variances)[-k:]
        selected_indices = np.sort(selected_indices)  # Sort for contiguous indexing
    else:
        # Select features above threshold
        selected_indices = np.where(variances > threshold)[0]
        
        if len(selected_indices) == 0:
            raise ValueError(
                f"No features have variance > {threshold}. "
                f"Max variance: {variances.max():.6f}, min: {variances.min():.6f}"
            )
    
    return selected_indices, variances


def select_mutual_info(
    data: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Select features using mutual information with target labels.
    
    Computes mutual information between each feature and the target,
    then selects top-k features.
    
    :param data: Feature matrix of shape (n_samples, n_features).
    :param labels: Labels of shape (n_samples,).
    :param k: Number of features to select.
    :param random_state: Random state for reproducibility.
    :return: Tuple of (selected_indices, mi_scores).
    """
    n_samples, n_features = data.shape
    
    # Compute mutual information
    mi_scores = mutual_info_classif(
        data,
        labels,
        discrete_features=False,
        random_state=random_state
    )
    
    # Select top-k features
    k = min(k, n_features)
    selected_indices = np.argsort(mi_scores)[-k:]
    selected_indices = np.sort(selected_indices)  # Sort for contiguous indexing
    
    return selected_indices, mi_scores


def select_l1(
    data: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None,
    C: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Select features using L1-regularized logistic regression.
    
    Trains a logistic regression model with L1 penalty and selects
    features with non-zero coefficients. Optionally selects top-k
    by absolute coefficient magnitude.
    
    :param data: Feature matrix of shape (n_samples, n_features).
    :param labels: Labels of shape (n_samples,).
    :param k: Optional number of features to select (top-k by |coefficient|).
    :param C: Inverse regularization strength (smaller = more regularization).
    :param random_state: Random state for reproducibility.
    :return: Tuple of (selected_indices, coefficients).
    """
    n_samples, n_features = data.shape
    
    # Standardize features for L1 regularization
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Train L1-regularized logistic regression
    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        random_state=random_state,
        max_iter=1000
    )
    model.fit(data_scaled, labels)
    
    # Get absolute coefficients
    if len(model.coef_.shape) == 1:
        coefficients = np.abs(model.coef_)
    else:
        # For binary classification, take first class coefficients
        coefficients = np.abs(model.coef_[0])
    
    if k is not None:
        # Select top-k by absolute coefficient
        k = min(k, n_features)
        selected_indices = np.argsort(coefficients)[-k:]
        selected_indices = np.sort(selected_indices)  # Sort for contiguous indexing
    else:
        # Select features with non-zero coefficients
        selected_indices = np.where(coefficients > 1e-10)[0]
        
        if len(selected_indices) == 0:
            # If all coefficients are zero, select top feature
            selected_indices = np.array([np.argmax(coefficients)])
    
    return selected_indices, coefficients


def select_pipeline(
    data: np.ndarray,
    labels: np.ndarray,
    stages: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Apply multiple feature selection methods in sequence.
    
    Each stage reduces the feature set, and indices are composed to map
    back to the original feature space.
    
    :param data: Feature matrix of shape (n_samples, n_features).
    :param labels: Labels of shape (n_samples,).
    :param stages: List of stage configurations, each with 'method' and method-specific params.
    :return: Tuple of (selected_indices_in_original, final_scores, stage_info).
        - selected_indices_in_original: Indices in original feature space
        - final_scores: Scores from the last stage (in original feature space)
        - stage_info: List of dicts with info about each stage
    """
    if not stages or len(stages) == 0:
        raise ValueError("Pipeline must have at least one stage.")
    
    current_data = data
    current_indices = np.arange(data.shape[1])  # Track mapping to original features
    stage_info = []
    
    for stage_idx, stage_config in enumerate(stages):
        stage_name = stage_config.get('name', f'stage_{stage_idx + 1}')
        method = stage_config.get('method')
        
        if method is None:
            raise ValueError(f"Stage {stage_idx + 1} ({stage_name}) must have 'method' key.")
        
        # Extract method-specific parameters
        stage_params = {k: v for k, v in stage_config.items() 
                       if k not in ['method', 'name']}
        
        print(f"  [Pipeline Stage {stage_idx + 1}/{len(stages)}] {stage_name}: {method} on {current_data.shape[1]} features")
        
        # Get k parameter
        k = stage_params.get('k', None)
        
        # Apply selection method
        method = method.lower()
        
        if method == "amgm":
            if k is None:
                raise ValueError(f"Stage {stage_name}: 'k' is required for AMGM method.")
            mode = stage_params.get('mode', 'ratio')
            selected_local, scores = select_amgm(current_data, labels, k, mode=mode)
            
        elif method in ["cosine", "cosine_redundancy"]:
            if k is None:
                raise ValueError(f"Stage {stage_name}: 'k' is required for cosine method.")
            threshold = stage_params.get('threshold', 0.95)
            sel_method = stage_params.get('selection_method', 'greedy')
            selected_local, scores = select_cosine_redundancy(
                current_data, k, threshold=threshold, method=sel_method
            )
            
        elif method in ["variance", "variance_threshold"]:
            threshold = stage_params.get('threshold', 0.0)
            selected_local, scores = select_variance_threshold(
                current_data, threshold=threshold, k=k
            )
            
        elif method in ["mutual_info", "mi"]:
            if k is None:
                raise ValueError(f"Stage {stage_name}: 'k' is required for mutual info method.")
            random_state = stage_params.get('random_state', 42)
            selected_local, scores = select_mutual_info(
                current_data, labels, k, random_state=random_state
            )
            
        elif method == "l1":
            C = stage_params.get('C', 1.0)
            random_state = stage_params.get('random_state', 42)
            selected_local, scores = select_l1(
                current_data, labels, k=k, C=C, random_state=random_state
            )
            
        else:
            raise ValueError(
                f"Stage {stage_name}: Unknown method '{method}'. "
                f"Available: amgm, cosine, variance, mutual_info, l1"
            )
        
        # Compose indices: map local selection back to original feature space
        selected_in_original = current_indices[selected_local]
        
        # Store stage information
        stage_info.append({
            'name': stage_name,
            'method': method,
            'features_in': current_data.shape[1],
            'features_out': len(selected_local),
            'selected_indices_local': selected_local,
            'selected_indices_original': selected_in_original,
            'scores': scores,
            'params': stage_params
        })
        
        print(f"    → Selected {len(selected_local)} features")
        
        # Update for next stage
        current_data = current_data[:, selected_local]
        current_indices = selected_in_original
    
    # Final indices and scores
    final_indices = current_indices
    
    # Create final scores array (in original space) from last stage
    final_scores_full = np.zeros(data.shape[1])
    final_scores_full[final_indices] = stage_info[-1]['scores'][stage_info[-1]['selected_indices_local']]
    
    return final_indices, final_scores_full, stage_info


def select_features(
    data: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    k: Optional[int] = None,
    **kwargs
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Unified interface for feature selection.
    
    :param data: Feature tensor of shape (n_samples, n_features).
    :param labels: Label tensor of shape (n_samples,).
    :param method: Selection method name.
    :param k: Number of features to select (required for most methods).
    :param kwargs: Additional method-specific parameters.
    :return: Tuple of (selected_data, selected_indices, scores).
    """
    # Convert to numpy
    data_np = data.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Call appropriate selection method
    method = method.lower()
    
    if method == "pipeline":
        # Multi-stage feature selection
        stages = kwargs.get('stages')
        if stages is None or not isinstance(stages, list):
            raise ValueError("Pipeline method requires 'stages' parameter (list of stage configs).")
        
        selected_indices, scores, stage_info = select_pipeline(data_np, labels_np, stages)
        
        # Store stage info in kwargs for caller to retrieve
        if '_stage_info_out' in kwargs:
            kwargs['_stage_info_out'].extend(stage_info)
        
    elif method == "amgm":
        if k is None:
            raise ValueError("Parameter 'k' is required for AMGM method.")
        mode = kwargs.get("mode", "ratio")
        selected_indices, scores = select_amgm(data_np, labels_np, k, mode=mode)
        
    elif method == "cosine" or method == "cosine_redundancy":
        if k is None:
            raise ValueError("Parameter 'k' is required for cosine redundancy method.")
        threshold = kwargs.get("threshold", 0.95)
        sel_method = kwargs.get("selection_method", "greedy")
        selected_indices, scores = select_cosine_redundancy(data_np, k, threshold=threshold, method=sel_method)
        
    elif method == "variance" or method == "variance_threshold":
        threshold = kwargs.get("threshold", 0.0)
        selected_indices, scores = select_variance_threshold(data_np, threshold=threshold, k=k)
        
    elif method == "mutual_info" or method == "mi":
        if k is None:
            raise ValueError("Parameter 'k' is required for mutual information method.")
        random_state = kwargs.get("random_state", 42)
        selected_indices, scores = select_mutual_info(data_np, labels_np, k, random_state=random_state)
        
    elif method == "l1":
        C = kwargs.get("C", 1.0)
        random_state = kwargs.get("random_state", 42)
        selected_indices, scores = select_l1(data_np, labels_np, k=k, C=C, random_state=random_state)
        
    else:
        raise ValueError(
            f"Unknown feature selection method: {method}. "
            f"Available methods: amgm, cosine, variance, mutual_info, l1, pipeline"
        )
    
    # Apply selection to data
    # Make a copy to ensure contiguous memory (avoid negative stride issues)
    selected_data = data[:, selected_indices].clone()
    
    return selected_data, selected_indices, scores
