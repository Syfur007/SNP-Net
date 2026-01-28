# Weighted Soft-Voting Ensemble for ASD SNP Classification

This document describes the implementation and usage of the weighted soft-voting ensemble system for combining predictions from multiple deep learning models trained on ASD SNP data.

## Overview

The ensemble system combines predictions from 5 pre-trained models using weighted soft voting:
- **Bi-LSTM**: Bidirectional LSTM with sequence modeling
- **Stacked LSTM**: Multi-layer hierarchical LSTM
- **GRU**: Gated Recurrent Unit
- **Autoencoder + MLP**: Autoencoder with classifier head
- **VAE + MLP**: Variational Autoencoder with classifier head

### Weighted Soft Voting Formula

For each test sample, the ensemble computes:

$$P_{final} = \sum_{i=1}^{N} w_i \cdot P_i$$

where:
- $w_i = \frac{Acc_i}{\sum_j Acc_j}$ (normalized validation accuracy weights)
- $P_i = [P_{control}, P_{ASD}]$ (probability distribution from model $i$)
- Final prediction: $\hat{y} = \arg\max(P_{final})$

## Features

✅ **Preprocessing Alignment**: Ensures identical normalization and feature selection (AMGM + cosine redundancy to 1000 SNPs) as during training

✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

✅ **Variance Analysis**: Computes cross-model standard deviation, agreement matrices, and variance reduction

✅ **Publication-Ready Plots**: ROC curves, metric comparisons, variance analysis, confusion matrices

✅ **Fully Modular**: Models and weights configurable via Hydra YAML

✅ **Checkpoint Auto-Discovery**: Helper script to find trained models and generate configurations

## Directory Structure

```
src/ensemble/
├── __init__.py                    # Package exports
├── voting_ensemble.py             # WeightedSoftVotingEnsemble class
├── preprocessing.py               # Preprocessing alignment utilities
├── ensemble_evaluator.py          # Evaluation and metrics computation
└── plotting.py                    # Visualization utilities

configs/ensemble/
├── autism_voting.yaml             # Ensemble config for autism dataset
└── autism_voting_auto.yaml        # Auto-generated config (after discovery)

tools/
└── discover_ensemble_checkpoints.py  # Auto-discover trained models

src/
└── ensemble_eval.py               # Main ensemble evaluation script
```

## Installation

No additional dependencies required beyond the base SNP-Net environment.

## Usage

### Step 1: Train Individual Models

First, train all required models on the autism dataset:

```bash
# Train Bi-LSTM
python src/train.py experiment=autism_bilstm

# Train Stacked LSTM
python src/train.py experiment=autism_stacked_lstm

# Train GRU
python src/train.py experiment=autism_gru

# Train Autoencoder
python src/train.py experiment=autism_autoencoder

# Train VAE
python src/train.py experiment=autism_vae
```

**Note:** Training will save checkpoints to `logs/train/runs/{experiment_name}_{timestamp}/checkpoints/`. The best checkpoint is automatically selected based on validation accuracy.

### Step 2: Auto-Discover Checkpoints (Recommended)

Use the checkpoint discovery script to automatically find trained models and generate ensemble configuration:

```bash
python tools/discover_ensemble_checkpoints.py \
    --dataset autism \
    --models bilstm stacked_lstm gru autoencoder vae \
    --output configs/ensemble/autism_voting_auto.yaml
```

This will:
1. Scan `logs/train/runs/` and `logs/train/multiruns/` for trained model checkpoints
2. Match checkpoints by experiment name pattern (e.g., `autism_bilstm_*`)
3. Extract validation accuracies from checkpoint metadata (from `ModelCheckpoint` callback state)
4. Generate a properly configured YAML file with absolute checkpoint paths
5. Auto-detect GPU availability and set appropriate device

**How checkpoint discovery works:**
- Searches for directories matching `{dataset}_{model}_*` pattern
- Falls back to checking `.hydra/config.yaml` for older directory structures
- Prioritizes `best.ckpt` if available, otherwise selects checkpoint with highest epoch
- Extracts validation accuracy from checkpoint's callback state

**Output example:**
```
Searching for bilstm checkpoints...
  Found matching directory: autism_bilstm_2026-01-12_14-30-22
  ✓ Found checkpoint: logs/train/runs/autism_bilstm_2026-01-12_14-30-22/checkpoints/best.ckpt
    Validation accuracy: 0.8523

Searching for stacked_lstm checkpoints...
  Found matching directory: autism_stacked_lstm_2026-01-12_15-45-10
  ✓ Found checkpoint: logs/train/runs/autism_stacked_lstm_2026-01-12_15-45-10/checkpoints/best.ckpt
    Validation accuracy: 0.8431
...

✓ Found 5/5 model checkpoints.
✓ Generated ensemble configuration: configs/ensemble/autism_voting_auto.yaml

To run ensemble evaluation:
  python src/ensemble_eval.py ensemble=autism_voting_auto
```

### Step 3: Run Ensemble Evaluation

Evaluate the ensemble on the test set:

```bash
# Using auto-generated config (recommended)
python src/ensemble_eval.py ensemble=autism_voting_auto

# Or using manually configured file
python src/ensemble_eval.py ensemble=autism_voting
```

**What happens:**
1. Loads all model checkpoints specified in the config
2. Verifies preprocessing consistency across models
3. Applies identical preprocessing to test data as used during training
4. Computes weighted soft-voting predictions using validation accuracy weights
5. Evaluates ensemble and individual models on test set
6. Generates comprehensive metrics and visualizations

### Step 4: View Results

Results are saved to `logs/ensemble_eval/runs/{config_name}_{timestamp}/ensemble_results/`:

Example: `logs/ensemble_eval/runs/autism_voting_auto_2026-01-12_10-30-45/ensemble_results/`

```
ensemble_results/
├── model_comparison.csv           # Comparison table (individual vs ensemble)
├── ensemble_metrics.txt           # Detailed ensemble metrics
├── variance_analysis.txt          # Variance and agreement analysis
├── predictions.npz                # Predictions and probabilities
├── confusion_matrix.csv           # Confusion matrix
└── plots/
    ├── roc_comparison.png         # ROC curves (individual + ensemble)
    ├── metric_comparison.png      # Bar charts of metrics
    ├── variance_analysis.png      # Agreement heatmap + accuracy distribution
    └── confusion_matrix.png       # Ensemble confusion matrix
```

The directory name includes:
- **config_name**: The ensemble config used (e.g., `autism_voting_auto`)
- **timestamp**: Execution time in format `YYYY-MM-DD_HH-MM-SS`

## Configuration

### Ensemble Configuration Format

```yaml
# configs/ensemble/autism_voting.yaml
ensemble:
  weighting_strategy: "accuracy"  # or "custom"
  device: "cuda"                  # or "cpu"
  save_predictions: true
  generate_plots: true
  
  models:
    - name: "bilstm"
      checkpoint_path: "logs/train/runs/.../checkpoints/best.ckpt"
      val_accuracy: 0.85
    
    - name: "stacked_lstm"
      checkpoint_path: "logs/train/runs/.../checkpoints/best.ckpt"
      val_accuracy: 0.83
    
    # ... more models
```

### Manual Configuration

To manually specify checkpoint paths and validation accuracies:

1. Copy the template: `cp configs/ensemble/autism_voting.yaml configs/ensemble/autism_voting_custom.yaml`
2. Edit `configs/ensemble/autism_voting_custom.yaml`
3. Update `checkpoint_path` for each model with absolute or workspace-relative paths
4. Update `val_accuracy` for each model (find these in training logs or checkpoint metadata)
5. Use it: `python src/ensemble_eval.py ensemble=autism_voting_custom`

**Finding validation accuracies manually:**
```bash
# Check training logs
cat logs/train/runs/{experiment_run}/train.log | grep "val/acc"

# Or use the discovery script to see found accuracies
python tools/discover_ensemble_checkpoints.py --dataset autism
```

### Command-Line Overrides

Override configuration via command line:

```bash
# Use CPU instead of GPU
python src/ensemble_eval.py ensemble=autism_voting ensemble.device=cpu

# Disable plot generation
python src/ensemble_eval.py ensemble=autism_voting ensemble.generate_plots=false

# Change batch size
python src/ensemble_eval.py ensemble=autism_voting data.batch_size=64
```

## Advanced Usage

### Adding/Removing Models

To exclude a model from the ensemble, simply remove it from the `models` list in the config:

```yaml
ensemble:
  models:
    - name: "bilstm"
      checkpoint_path: "..."
      val_accuracy: 0.85
    
    - name: "gru"
      checkpoint_path: "..."
      val_accuracy: 0.84
    
    # VAE excluded - only using 2 models
```

### Custom Weighting

To use custom weights instead of validation accuracy:

```yaml
ensemble:
  weighting_strategy: "custom"
  
  models:
    - name: "bilstm"
      checkpoint_path: "..."
      weight: 0.4  # Custom weight
    
    - name: "gru"
      checkpoint_path: "..."
      weight: 0.6  # Custom weight
```

### Mental Health Dataset

To run ensemble on the mental health dataset:

```bash
# Train mental health models first
python src/train.py experiment=mental_bilstm
python src/train.py experiment=mental_stacked_lstm
python src/train.py experiment=mental_gru
python src/train.py experiment=mental_autoencoder
python src/train.py experiment=mental_vae

# Discover mental health checkpoints
python tools/discover_ensemble_checkpoints.py \
    --dataset mental \
    --output configs/ensemble/mental_voting_auto.yaml

# Run evaluation
python src/ensemble_eval.py ensemble=mental_voting_auto
```

## Output Interpretation

### Model Comparison Table

```
Model          Type        Accuracy  Precision  Recall  F1-Score  ROC-AUC  Improvement (%)
bilstm         Individual  0.8500    0.8600     0.8400  0.8500    0.9100   0.00
stacked_lstm   Individual  0.8300    0.8400     0.8200  0.8300    0.8900   0.00
gru            Individual  0.8400    0.8500     0.8300  0.8400    0.9000   0.00
autoencoder    Individual  0.8200    0.8300     0.8100  0.8200    0.8800   0.00
vae            Individual  0.8100    0.8200     0.8000  0.8100    0.8700   0.00
Ensemble       Ensemble    0.8700    0.8800     0.8600  0.8700    0.9300   2.35
```

**Improvement (%)**: Percentage improvement of ensemble over best individual model.

### Variance Analysis

```
Mean Accuracy: 0.8300
Std Accuracy: 0.0141
Min Accuracy: 0.8100
Max Accuracy: 0.8500
Ensemble Accuracy: 0.8700
Mean Agreement: 0.8945
```

- **Std Accuracy**: Standard deviation across individual models (lower = more consistent)
- **Mean Agreement**: Average pairwise prediction agreement (higher = models agree more)
- **Variance Reduction**: Ensemble stability improvement over best individual model

### ROC Curves

The ROC comparison plot shows:
- Individual model ROC curves with AUC values
- Ensemble ROC curve (highlighted in dark red)
- Random classifier baseline (diagonal dashed line)

**Ideal**: Ensemble curve should be above or equal to all individual curves.

## Troubleshooting

### Error: "Checkpoint not found"

**Cause**: Invalid checkpoint path in configuration.

**Solution**: 
1. Run checkpoint discovery script: `python tools/discover_ensemble_checkpoints.py --dataset autism`
2. Or manually verify checkpoint paths exist: `ls logs/train/runs/*/checkpoints/best.ckpt`
3. Check that experiment names match: training creates directories like `autism_bilstm_{timestamp}`

### Error: "Preprocessing parameters inconsistent"

**Cause**: Models trained with different data preprocessing settings.

**Solution**: 
- Ensure all models were trained with identical `data` configuration
- Especially check `feature_selection` settings in training configs
- If intentional, set `ensemble.allow_inconsistent_preprocessing=true` (not recommended)

### Error: "CUDA out of memory"

**Cause**: GPU memory insufficient for loading all models.

**Solution**:
```bash
# Use CPU
python src/ensemble_eval.py ensemble=autism_voting ensemble.device=cpu

# Or reduce batch size
python src/ensemble_eval.py ensemble=autism_voting data.batch_size=16
```

### Warning: "Validation accuracies > 1.0 detected"

**Cause**: Validation accuracies entered as percentages (e.g., 85 instead of 0.85).

**Solution**: Ensemble will auto-correct by dividing by 100. Update config for clarity.

## Performance Expectations

Based on typical ensemble behavior:

- **Accuracy Improvement**: 1-3% over best individual model
- **Variance Reduction**: 20-40% reduction in prediction variance
- **Inference Time**: ~5x slower than single model (5 forward passes)
- **Memory Usage**: ~5x single model memory (all models loaded)

## API Reference

### WeightedSoftVotingEnsemble

```python
from src.ensemble import WeightedSoftVotingEnsemble

ensemble = WeightedSoftVotingEnsemble(
    checkpoint_paths={
        "bilstm": "path/to/bilstm.ckpt",
        "gru": "path/to/gru.ckpt",
    },
    validation_accuracies={
        "bilstm": 0.85,
        "gru": 0.84,
    },
    device="cuda",
    num_classes=2,
)

# Predict
preds, probs, individual_probs = ensemble.predict(x)
```

### EnsembleEvaluator

```python
from src.ensemble import EnsembleEvaluator

evaluator = EnsembleEvaluator(
    ensemble=ensemble,
    task="binary",
    num_classes=2,
)

results = evaluator.evaluate(dataloader)
evaluator.print_summary()
evaluator.save_results(output_dir)
```

## Citation

If you use this ensemble system in your research, please cite:

```bibtex
@software{snp_net_ensemble,
  title={Weighted Soft-Voting Ensemble for SNP Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/Syfur007/SNP-Net}
}
```

## Contributing

To extend the ensemble system:

1. **Add new ensemble strategies**: Implement in `src/ensemble/` (e.g., stacking, boosting)
2. **Add new weighting schemes**: Modify `WeightedSoftVotingEnsemble._compute_weights()`
3. **Add new metrics**: Extend `EnsembleEvaluator._compute_metrics()`
4. **Add new plots**: Add functions to `src/ensemble/plotting.py`

## License

Same license as SNP-Net project.
