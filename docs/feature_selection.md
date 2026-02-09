# Feature Selection

Feature selection in SNP-Net enables dimensionality reduction by selecting the most informative features from your SNP data. The system supports both single-stage and multi-stage pipeline selection methods.

## Table of Contents

- [Quick Start](#quick-start)
- [Available Methods](#available-methods)
- [Multi-Stage Pipelines](#multi-stage-pipelines)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Implementation Details](#implementation-details)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Quick Start

Enable feature selection by adding configuration to your data config file:

```yaml
# configs/data/autism.yaml
feature_selection:
  method: mutual_info
  k: 100
  random_state: 42
```

Run training:
```bash
python src/train.py data=autism
```

Feature selection is integrated into the `DataModule` and applied during the `setup()` phase, before dataset creation and splitting. This ensures:
- All data splits use the same selected features
- Selected features are saved in checkpoints for reproducible inference
- No code changes required to enable/disable

### Exported Artifacts

When training runs, the active feature-selection pipeline automatically exports structured outputs to
`logs/.../exports/fold_*/seed_*/`:

- `selected_snps.csv` (final SNP list with ranks and scores)
- `feature_scores.npy` (full score vector in original feature space)
- `feature_pipeline_stages.json` and `feature_pipeline_summary.csv` (pipeline metadata)
- `preprocessing_summary.json` (feature counts before/after)
- `snp_missingness_maf.csv` (missingness + MAF for each SNP)
- `dataset_metadata.json` (split indices, class ratios, and seeds)

## Available Methods

SNP-Net provides five feature selection methods plus multi-stage pipeline support:

### 1. AMGM (Arithmetic-Geometric Mean Ratio)

**Type**: Filter method (univariate) | **Complexity**: O(n·f) | **Best for**: Binary classification

Ranks features by the ratio difference of arithmetic to geometric mean between classes.

```yaml
feature_selection:
  method: amgm
  k: 100
  mode: ratio  # or 'diff'
```

**Parameters**:
- `k` (required): Number of features to select
- `mode` (default: `"ratio"`): `"ratio"` for AM/GM ratio, `"diff"` for mean difference

**When to use**: Fast selection for binary classification problems with interpretable feature scoring.

---

### 2. Cosine Redundancy

**Type**: Filter method (pairwise) | **Complexity**: O(f²·k) | **Best for**: Removing correlated features

Removes redundant features based on pairwise cosine similarity.

```yaml
feature_selection:
  method: cosine
  k: 100
  threshold: 0.95
  selection_method: greedy  # or 'cluster'
```

**Parameters**:
- `k` (required): Target number of features
- `threshold` (default: `0.95`): Similarity threshold for redundancy
- `selection_method` (default: `"greedy"`): `"greedy"` or `"cluster"`

**When to use**: High feature correlation in data (common in SNP datasets), after pre-filtering to < 5000 features.

**Warning**: Memory intensive (O(f²)) for large feature counts. Use univariate methods first.

---

### 3. Variance Threshold

**Type**: Filter method (univariate) | **Complexity**: O(n·f) | **Best for**: Pre-filtering

Selects features based on variance across samples.

```yaml
feature_selection:
  method: variance
  k: 100  # or use threshold: 0.01
```

**Parameters**:
- `k` (optional): Number of features to select
- `threshold` (optional): Minimum variance threshold

**When to use**: Quick baseline, removing constant/near-constant features, initial pre-processing.

---

### 4. Mutual Information

**Type**: Filter method (univariate) | **Complexity**: O(n·f·log n) | **Best for**: General purpose

Ranks features by mutual information with target labels. Captures non-linear relationships.

```yaml
feature_selection:
  method: mutual_info
  k: 100
  random_state: 42
```

**Parameters**:
- `k` (required): Number of features to select
- `random_state` (default: `42`): Random seed for reproducibility

**When to use**: Recommended default choice. Works well for most classification tasks.

---

### 5. L1 Regularization

**Type**: Embedded method | **Complexity**: O(n·f·i) | **Best for**: Model-based selection

Uses L1-regularized logistic regression to select features with non-zero coefficients.

```yaml
feature_selection:
  method: l1
  k: 100
  C: 1.0  # smaller = more regularization
  random_state: 42
```

**Parameters**:
- `k` (optional): Number of features to select (top-k by coefficient magnitude)
- `C` (default: `1.0`): Inverse regularization strength
- `random_state` (default: `42`): Random seed

**When to use**: Model-based selection with linear relationships, controlling sparsity via regularization.

---

## Multi-Stage Pipelines

For very high-dimensional data (100,000+ features), use multi-stage pipelines to efficiently reduce features in sequential stages.

### Pipeline Configuration

```yaml
feature_selection:
  method: pipeline
  stages:
    - name: initial_filter
      method: variance
      k: 5000
    - name: informative_selection
      method: amgm
      k: 1000
      mode: ratio
    - name: redundancy_removal
      method: cosine
      k: 100
      threshold: 0.95
```

### How It Works

1. Each stage applies a selection method to the current feature set
2. Indices are automatically composed to map back to original feature space
3. All stage information is logged and saved in checkpoints
4. Memory efficient: only stores current reduced data between stages

### Pipeline Example

**Scenario**: Reduce from 225,783 features to 1,000

```yaml
feature_selection:
  method: pipeline
  stages:
    - name: initial_amgm
      method: amgm
      k: 5000
      mode: ratio
    - name: remove_redundancy
      method: cosine
      k: 1000
      threshold: 0.95
```

**Result**:
```
[Pipeline Stage 1/2] initial_amgm: amgm on 225783 features
  → Selected 5000 features
[Pipeline Stage 2/2] remove_redundancy: cosine on 5000 features
  → Selected 1000 features
✓ Feature selection complete. Selected 1000 features
```

### Best Practices for Pipelines

1. **Start with fast methods**: Use variance or AMGM for initial filtering
2. **Progressive reduction**: Reduce gradually (100k → 10k → 1k → 100)
3. **Expensive methods last**: Apply cosine redundancy on smaller feature sets
4. **Name your stages**: Use descriptive names for better logging and debugging

## Configuration

### Enable Feature Selection

Add to data config file (`configs/data/your_data.yaml`):

```yaml
feature_selection:
  method: mutual_info
  k: 100
```

### Disable Feature Selection

```yaml
feature_selection: null
```

### Experiment Configuration

Pipeline in experiment config (`configs/experiment/your_experiment.yaml`):

```yaml
data:
  feature_selection:
    method: pipeline
    stages:
      - name: variance_filter
        method: variance
        k: 5000
      - name: mi_selection
        method: mutual_info
        k: 100
        random_state: 42
```

## Usage Examples

### Single-Stage Selection

```python
from src.data.datamodule import DataModule

# Configure datamodule with feature selection
dm = DataModule(
    data_file="data/FinalizedAutismData.csv",
    feature_selection={
        "method": "amgm",
        "k": 100,
        "mode": "ratio"
    }
)

dm.setup()
print(f"Selected {dm.num_features} features")
print(f"Indices: {dm._selected_indices}")
```

### Multi-Stage Pipeline

```python
dm = DataModule(
    data_file="data/FinalizedAutismData.csv",
    feature_selection={
        "method": "pipeline",
        "stages": [
            {"name": "stage1", "method": "variance", "k": 5000},
            {"name": "stage2", "method": "amgm", "k": 1000, "mode": "ratio"},
            {"name": "stage3", "method": "cosine", "k": 100}
        ]
    }
)

dm.setup()

# Inspect pipeline stages
for i, stage in enumerate(dm._feature_stages):
    print(f"Stage {i+1} ({stage['name']}): "
          f"{stage['features_in']} → {stage['features_out']} features")
```

### Direct API Usage

```python
from src.data.feature_selection import select_features
import torch

data = torch.randn(100, 1000)  # 100 samples, 1000 features
labels = torch.randint(0, 2, (100,))

# Single-stage
selected_data, indices, scores = select_features(
    data, labels, 
    method="mutual_info", 
    k=100,
    random_state=42
)

# Pipeline
stage_info = []
selected_data, indices, scores = select_features(
    data, labels,
    method="pipeline",
    stages=[
        {"method": "variance", "k": 500},
        {"method": "mutual_info", "k": 100}
    ],
    _stage_info_out=stage_info
)
```

### Inspect Selected Features from Checkpoint

```python
import torch

ckpt = torch.load("path/to/checkpoint.ckpt")
dm_state = ckpt['datamodule_state']

# Single-stage selection
if 'selected_indices' in dm_state:
    indices = dm_state['selected_indices']
    scores = dm_state['feature_scores']
    print(f"Selected {len(indices)} features")
    print(f"Top 10 indices: {indices[:10]}")

# Pipeline selection
if 'feature_stages' in dm_state:
    stages = dm_state['feature_stages']
    print(f"\nPipeline: {len(stages)} stages")
    for i, stage in enumerate(stages):
        print(f"  Stage {i+1} ({stage['name']}): "
              f"{stage['features_in']} → {stage['features_out']}")
```

## Implementation Details

### Architecture

```
User Config (YAML)
      ↓
DataModule.__init__()
      ↓
   setup()  ← Called by Lightning Trainer
      ↓
_select_features()  ← Checks if feature_selection config exists
      ↓
feature_selection.select_features()  ← Routes to appropriate method
      ↓
Individual method (AMGM, MI, etc.) or Pipeline
      ↓
Returns: (selected_data, indices, scores)
      ↓
DataModule stores: _selected_indices, _feature_scores, _feature_stages
      ↓
state_dict() / load_state_dict()  ← Persists to checkpoint
```

### Data Flow

1. **Initialization**: DataModule receives `feature_selection` config from Hydra
2. **Setup**: When `setup()` is called, loads data from CSV
3. **Selection**: Calls `_select_features()` if config is present
4. **Index Composition**: Pipeline stages compose indices to map to original space
5. **Dataset Creation**: Creates train/val/test datasets with selected features
6. **Checkpoint Persistence**: Saves indices, scores, and pipeline metadata via `state_dict()`

### Key Features

- **Lazy Execution**: Selection happens at `setup()`, not `__init__()` (efficient for distributed training)
- **Checkpoint Persistence**: All selection metadata saved in Lightning checkpoints
- **Index Composition**: Multi-stage pipelines correctly map final indices to original feature space
- **Stage Metadata**: Pipeline stages logged with features_in, features_out, method, params
- **Reproducibility**: Random seeds propagated through all selection methods

## Performance Considerations

### Computational Complexity

| Method | Time Complexity | Space Complexity | Recommended Max Features |
|--------|----------------|------------------|--------------------------|
| Variance | O(n·f) | O(f) | 1M+ |
| AMGM | O(n·f) | O(f) | 500K |
| Mutual Info | O(n·f·log n) | O(f) | 100K |
| L1 | O(n·f·i) | O(f) | 50K |
| Cosine | O(f²·k) | O(f²) | 5K |
| Pipeline | Sum of stages | Max of stages | 1M+ (with staging) |

*where n=samples, f=features, k=target features, i=iterations*

### Memory Optimization

**Problem**: 500,000 features × 1,000 samples = 500M entries (~2GB for float32)

**Solutions**:
1. **Use pipelines**: Reduce features progressively
2. **Pre-filter with variance**: Quick reduction before expensive methods
3. **Batch processing**: For cosine similarity, process in chunks
4. **Early stopping**: Use fewer features in early stages

### Performance Tips

1. **Pipeline Strategy for Large-Scale Data**:
   ```yaml
   # For 100K+ features, use this pattern:
   stages:
     - {method: variance, k: 10000}    # Fast: ~1s
     - {method: amgm, k: 1000}         # Medium: ~5s
     - {method: cosine, k: 100}        # Slow but on small set: ~2s
   ```

2. **Single-Stage for Medium Data**:
   ```yaml
   # For <50K features:
   feature_selection:
     method: mutual_info
     k: 100
   ```

3. **Avoid**:
   - Cosine on >10K features (O(f²) memory)
   - L1 without pre-filtering on >50K features
   - More than 5 pipeline stages (diminishing returns)

### Benchmark (on autism dataset, 567 samples × 225,783 features)

**Pipeline: variance(5K) → amgm(1K) → cosine(100)**
- Stage 1: 2.3s (variance)
- Stage 2: 1.8s (amgm)
- Stage 3: 3.1s (cosine)
- **Total**: ~7 seconds

**Single-stage: amgm(1K) on full data**
- **Total**: ~15 seconds

**Speedup**: 2.1x with pipeline (and produces different, potentially better features)

## Troubleshooting

### Common Issues

#### 1. `ValueError: Pipeline method requires 'stages' parameter`

**Cause**: Missing or incorrectly formatted `stages` in config

**Solution**:
```yaml
# ✓ Correct
feature_selection:
  method: pipeline
  stages:
    - name: stage1
      method: amgm
      k: 1000

# ✗ Wrong
feature_selection:
  method: pipeline
  k: 1000  # Missing stages!
```

#### 2. `RuntimeError: selected index k out of range`

**Cause**: Requesting more features than available

**Solution**: Ensure each stage's `k` is less than input features
```yaml
# If you have 1000 features:
stages:
  - {method: variance, k: 500}   # ✓ OK
  - {method: amgm, k: 1500}      # ✗ Error! 1500 > 500
```

#### 3. Memory Error with Cosine Similarity

**Cause**: Cosine requires O(f²) memory

**Solution**: Pre-filter to <5000 features before cosine
```yaml
stages:
  - {method: variance, k: 5000}  # Pre-filter
  - {method: cosine, k: 100}     # Now safe
```

#### 4. Indices Not Persisted in Checkpoint

**Cause**: Selection happened after `state_dict()` was called

**Solution**: Ensure `setup()` is called before saving checkpoint (Lightning handles this automatically)

#### 5. Different Results Across Runs

**Cause**: Non-deterministic methods (mutual_info, l1)

**Solution**: Set `random_state` parameter
```yaml
feature_selection:
  method: mutual_info
  k: 100
  random_state: 42  # Ensures reproducibility
```

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.getLogger("src.data.feature_selection").setLevel(logging.DEBUG)
   ```

2. **Inspect selection metadata**:
   ```python
   dm = DataModule(...)
   dm.setup()
   
   print(f"Selected indices: {dm._selected_indices}")
   print(f"Feature scores: {dm._feature_scores}")
   print(f"Pipeline stages: {dm._feature_stages}")
   ```

3. **Validate pipeline composition**:
   ```python
   # After setup, check that final indices are valid
   assert len(dm._selected_indices) == dm.num_features
   assert all(0 <= idx < original_feature_count for idx in dm._selected_indices)
   ```

4. **Check checkpoint contents**:
   ```python
   ckpt = torch.load("checkpoint.ckpt")
   print(ckpt['datamodule_state'].keys())
   # Should contain: selected_indices, feature_scores, feature_stages (if pipeline)
   ```

## API Reference

### Main Interface

#### `select_features(data, labels, method, **kwargs) -> (Tensor, Tensor, Optional[Tensor])`

Unified interface for all feature selection methods.

**Parameters**:
- `data` (Tensor): Input features of shape (n_samples, n_features)
- `labels` (Tensor): Target labels of shape (n_samples,)
- `method` (str): Selection method (`"amgm"`, `"cosine"`, `"variance"`, `"mutual_info"`, `"l1"`, `"pipeline"`)
- `**kwargs`: Method-specific parameters

**Returns**:
- `selected_data` (Tensor): Reduced feature set (n_samples, k)
- `indices` (Tensor): Selected feature indices in original space
- `scores` (Optional[Tensor]): Feature importance scores (None for pipeline/cosine)

**Example**:
```python
from src.data.feature_selection import select_features

selected_data, indices, scores = select_features(
    data, labels,
    method="amgm",
    k=100,
    mode="ratio"
)
```

---

### Individual Methods

#### `select_amgm(data, labels, k, mode="ratio") -> (Tensor, Tensor, Tensor)`

AMGM-based univariate feature ranking.

#### `select_cosine_redundancy(data, k, threshold=0.95, selection_method="greedy") -> (Tensor, Tensor)`

Remove redundant features based on cosine similarity.

#### `select_variance_threshold(data, k=None, threshold=None) -> (Tensor, Tensor, Tensor)`

Select features by variance (requires `k` or `threshold`).

#### `select_mutual_info(data, labels, k, random_state=42) -> (Tensor, Tensor, Tensor)`

Mutual information-based selection (handles non-linear relationships).

#### `select_l1_regularization(data, labels, k=None, C=1.0, random_state=42) -> (Tensor, Tensor, Tensor)`

L1-regularized logistic regression for embedded selection.

---

### Pipeline

#### `select_pipeline(data, labels, stages, _stage_info_out=None) -> (Tensor, Tensor, None)`

Multi-stage sequential feature selection with index composition.

**Parameters**:
- `data` (Tensor): Input features
- `labels` (Tensor): Target labels
- `stages` (List[Dict]): List of stage configurations, each with `name`, `method`, and method-specific params
- `_stage_info_out` (Optional[List]): If provided, will be filled with stage metadata

**Returns**:
- `selected_data` (Tensor): Final reduced features
- `indices` (Tensor): Composed indices mapping to original feature space
- `scores` (None): Not available for pipeline

**Example**:
```python
stage_info = []
selected_data, indices, _ = select_pipeline(
    data, labels,
    stages=[
        {"name": "stage1", "method": "variance", "k": 1000},
        {"name": "stage2", "method": "mutual_info", "k": 100, "random_state": 42}
    ],
    _stage_info_out=stage_info
)

# Inspect stages
for stage in stage_info:
    print(f"{stage['name']}: {stage['features_in']} → {stage['features_out']}")
```

---

### DataModule Integration

The `DataModule` class automatically integrates feature selection when configured:

```python
from src.data.datamodule import DataModule

dm = DataModule(
    data_file="data/FinalizedAutismData.csv",
    feature_selection={"method": "amgm", "k": 100}
)
dm.setup()

# Access selected metadata
print(dm._selected_indices)  # Tensor of selected feature indices
print(dm._feature_scores)    # Tensor of feature scores (if available)
print(dm._feature_stages)    # List of stage metadata (if pipeline)
```

**Checkpoint Persistence**: All selection metadata is automatically saved/loaded via Lightning checkpoints through `state_dict()` and `load_state_dict()`.

---

## Quick Reference

### Method Selection Guide

| Scenario | Recommended Method | Config Example |
|----------|-------------------|----------------|
| Binary classification, interpretable | AMGM | `{method: amgm, k: 100}` |
| General purpose | Mutual Info | `{method: mutual_info, k: 100}` |
| Remove correlations | Cosine | `{method: cosine, k: 100, threshold: 0.95}` |
| Model-based | L1 | `{method: l1, k: 100, C: 1.0}` |
| Pre-filtering | Variance | `{method: variance, k: 5000}` |
| 100K+ features | Pipeline | See pipeline examples above |

### Common Pipeline Patterns

**High-dimensional to low-dimensional (100K → 100)**:
```yaml
stages:
  - {name: pre_filter, method: variance, k: 5000}
  - {name: select_informative, method: mutual_info, k: 500}
  - {name: remove_redundancy, method: cosine, k: 100}
```

**Fast binary classification pipeline**:
```yaml
stages:
  - {name: amgm_filter, method: amgm, k: 1000, mode: ratio}
  - {name: final_selection, method: cosine, k: 100, threshold: 0.9}
```

**Model-based refinement**:
```yaml
stages:
  - {name: variance_filter, method: variance, k: 10000}
  - {name: l1_selection, method: l1, k: 100, C: 0.1}
```

---

## Further Information

For implementation details and source code, see:
- **Feature Selection Module**: `src/data/feature_selection.py`
- **DataModule Integration**: `src/data/datamodule.py`
- **Test Suite**: `test_feature_selection.py`
- **Example Scripts**: `example_feature_selection.py`, `example_pipeline_selection.py`
