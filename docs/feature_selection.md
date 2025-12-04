# Feature Selection for SNP-Net

This document describes the feature selection functionality implemented in SNP-Net, which allows you to reduce the dimensionality of your SNP data by selecting the most informative features.

## Overview

Feature selection is integrated directly into the `DataModule` and is applied during the `setup()` phase, before dataset creation and train/val/test splitting. This ensures:
- All data splits use the same selected features
- Selected features are saved in model checkpoints for reproducible inference
- Minimal code changes required to enable/disable feature selection

## Available Methods

### 1. AMGM (Arithmetic-Geometric Mean Ratio)
**Type**: Filter method (univariate)

Computes the ratio of arithmetic mean to geometric mean for each class, then ranks features by the difference between class ratios. Features with higher discrimination between classes are selected.

**Parameters**:
- `k`: Number of features to select (required)
- `mode`: Selection mode (default: `"ratio"`)
  - `"ratio"`: Use AM/GM ratio difference
  - `"diff"`: Use simple mean difference

**Example**:
```yaml
feature_selection:
  method: amgm
  k: 100
  mode: ratio
```

### 2. Cosine Redundancy
**Type**: Filter method (pairwise)

Removes redundant features based on cosine similarity. Features that are highly correlated with others are removed, keeping only the most representative ones.

**Parameters**:
- `k`: Target number of features to select (required)
- `threshold`: Cosine similarity threshold (default: `0.95`)
- `selection_method`: Method for selecting representatives (default: `"greedy"`)
  - `"greedy"`: Iteratively remove features with highest mean similarity
  - `"cluster"`: Cluster features and keep one per cluster

**Example**:
```yaml
feature_selection:
  method: cosine
  k: 100
  threshold: 0.95
  selection_method: greedy
```

### 3. Variance Threshold
**Type**: Filter method (univariate)

Selects features based on variance. Can either remove low-variance features or select top-k by variance.

**Parameters**:
- `k`: Number of features to select (optional)
- `threshold`: Minimum variance threshold (default: `0.0`)

**Example**:
```yaml
# Select top 100 features by variance
feature_selection:
  method: variance
  k: 100

# Or remove features with variance < 0.1
feature_selection:
  method: variance
  threshold: 0.1
```

### 4. Mutual Information
**Type**: Filter method (univariate)

Ranks features by mutual information with the target labels. Features with higher mutual information are more informative for classification.

**Parameters**:
- `k`: Number of features to select (required)
- `random_state`: Random seed for reproducibility (default: `42`)

**Example**:
```yaml
feature_selection:
  method: mutual_info
  k: 100
  random_state: 42
```

### 5. L1 Regularization
**Type**: Embedded method

Uses L1-regularized logistic regression to select features. Features with non-zero coefficients (or top-k by coefficient magnitude) are selected.

**Parameters**:
- `k`: Number of features to select (optional)
- `C`: Inverse regularization strength (default: `1.0`)
  - Smaller values = stronger regularization = fewer features
- `random_state`: Random seed for reproducibility (default: `42`)

**Example**:
```yaml
feature_selection:
  method: l1
  k: 100
  C: 1.0
  random_state: 42
```

## Usage

### Basic Usage

1. **Edit your data config file** (e.g., `configs/data/autism.yaml`):

```yaml
# ... other parameters ...

feature_selection:
  method: mutual_info
  k: 100
  random_state: 42
```

2. **Run training normally**:

```bash
python src/train.py data=autism
```

The feature selection will be applied automatically during data loading.

### Disable Feature Selection

Set `feature_selection: null` in your config:

```yaml
feature_selection: null
```

### Programmatic Usage

You can also use the feature selection methods directly in Python:

```python
from src.data.feature_selection import select_features
import torch

# Your data
data = torch.randn(100, 1000)  # 100 samples, 1000 features
labels = torch.randint(0, 2, (100,))  # Binary labels

# Apply feature selection
selected_data, selected_indices, scores = select_features(
    data, labels, 
    method="mutual_info", 
    k=100
)

print(f"Selected {selected_data.shape[1]} features")
print(f"Selected indices: {selected_indices}")
```

## Implementation Details

### Architecture

The feature selection is implemented in three main components:

1. **`src/data/feature_selection.py`**: Core selection algorithms
   - Individual methods: `select_amgm()`, `select_cosine_redundancy()`, etc.
   - Unified interface: `select_features()`

2. **`src/data/datamodule.py`**: Integration with DataModule
   - `_select_features()`: Wrapper that calls the appropriate method
   - Called in `setup()` after optional normalization
   - Saves selected indices in `state_dict()` for checkpointing

3. **Config files**: `configs/data/*.yaml`
   - `feature_selection` parameter with examples

### Data Flow

```
CSV Data → Load → (Normalize?) → Feature Selection → Dataset Creation → Train/Val/Test Split
```

### Checkpointing

Selected feature indices are automatically saved in model checkpoints:

```python
# Saved in checkpoint
state_dict = {
    'selected_indices': np.array([1, 5, 7, ...]),
    'feature_scores': np.array([0.8, 0.7, 0.6, ...]),
    'feature_selection_config': {'method': 'mutual_info', 'k': 100}
}
```

This ensures that:
- Inference uses the same features as training
- You can inspect which features were selected
- Results are reproducible

### K-Fold Cross Validation

Feature selection is currently applied **globally** before fold splitting. This means:
- All folds use the same selected features
- Feature selection sees all train+val data (not fold-specific)

For fold-specific selection (e.g., for wrapper methods like RFE), you would need to modify `_apply_fold()` to run selection per-fold.

## Performance Considerations

### Computational Complexity

| Method | Time Complexity | Notes |
|--------|----------------|-------|
| AMGM | O(n × f) | Fast, suitable for large datasets |
| Variance | O(n × f) | Very fast |
| Mutual Info | O(n × f × log(n)) | Moderate, uses sklearn |
| L1 | O(n × f × iter) | Slower, depends on convergence |
| Cosine (greedy) | O(f² × k) | Slow for many features, use for f < 5000 |
| Cosine (cluster) | O(f² + f × log(f)) | Faster but needs more memory |

Where:
- n = number of samples
- f = number of features
- k = target number of features

### Memory Usage

- **Most methods**: O(n × f) for data storage
- **Cosine redundancy**: O(f²) for similarity matrix (can be large!)
- **Recommendation**: For f > 10,000, use univariate methods (AMGM, variance, mutual info) first to reduce to ~1000 features, then apply cosine redundancy if needed

## Method Selection Guidelines

### When to use each method:

1. **AMGM**: 
   - Binary classification
   - Want interpretable feature scoring
   - Need fast selection

2. **Cosine Redundancy**:
   - High feature correlation (common in SNP data)
   - Already reduced features to < 5000
   - Want diverse feature set

3. **Variance Threshold**:
   - Quick baseline
   - Remove constant/near-constant features
   - Pre-processing step before other methods

4. **Mutual Information**:
   - General-purpose, works well for most tasks
   - Captures non-linear relationships
   - Good default choice

5. **L1 Regularization**:
   - Want model-based selection
   - Linear relationships expected
   - Need to control sparsity with C parameter

### Recommended Pipeline

For typical SNP datasets with 10,000+ features:

1. **Stage 1**: Remove low-variance features
   ```yaml
   feature_selection:
     method: variance
     k: 1000  # or threshold: 0.01
   ```

2. **Stage 2**: Apply mutual information or AMGM
   ```yaml
   feature_selection:
     method: mutual_info
     k: 200
   ```

3. **Stage 3** (optional): Remove redundant features
   ```yaml
   feature_selection:
     method: cosine
     k: 100
     threshold: 0.95
   ```

## Testing

Run the test script to verify all methods work:

```bash
python test_feature_selection.py
```

Expected output:
```
============================================================
Testing Feature Selection Methods
============================================================
...
✓ AMGM selected 10 features
✓ Cosine redundancy selected 10 features
✓ Variance threshold selected 10 features
✓ Mutual information selected 10 features
✓ L1 regularization selected 10 features
============================================================
```

## Troubleshooting

### Issue: "Unknown label" error

**Cause**: Labels in CSV are not 'case'/'control' or numeric  
**Solution**: Ensure last row contains 'case', 'control', 0, or 1 values

### Issue: "No features have variance > threshold"

**Cause**: Threshold too high for your data  
**Solution**: Lower the threshold or use `k` parameter instead

### Issue: Feature selection is slow

**Cause**: Too many features for pairwise methods  
**Solution**: Use univariate methods first (AMGM, variance, mutual_info) to reduce features, then apply cosine redundancy

### Issue: Different results on each run

**Cause**: Some methods use randomization  
**Solution**: Set `random_state: 42` in your config

## Future Extensions

Potential improvements for future versions:

1. **Per-fold selection**: Run selection on training data only for each fold
2. **Recursive Feature Elimination (RFE)**: Wrapper method with iterative elimination
3. **Relief-F**: Considers feature interactions
4. **mRMR**: Minimum redundancy, maximum relevance
5. **Boruta**: All-relevant feature selection
6. **Multi-stage pipelines**: Chain multiple selection methods
7. **Feature importance visualization**: Plot selected features and their scores

## References

- Mutual Information: [sklearn.feature_selection.mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
- L1 Regularization: [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Feature Selection Guide: [sklearn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
