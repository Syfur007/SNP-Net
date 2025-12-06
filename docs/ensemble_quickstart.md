# Ensemble Quick Start Guide

This guide will help you quickly set up and run the weighted soft-voting ensemble for ASD SNP classification.

## Prerequisites

- All 5 models (Bi-LSTM, Stacked LSTM, GRU, Autoencoder, VAE) trained on the autism dataset
- Models saved with ModelCheckpoint callback in Hydra experiment runs

## Quick Start (3 Steps)

### 1. Discover Trained Models

Run the checkpoint discovery script to automatically find your trained models:

```bash
python scripts/discover_ensemble_checkpoints.py \
    --dataset autism \
    --models bilstm stacked_lstm gru autoencoder vae
```

**Output**: Creates `configs/ensemble/autism_voting_auto.yaml` with discovered checkpoints and validation accuracies.

**Example output**:
```
======================================================================
ENSEMBLE CHECKPOINT DISCOVERY
======================================================================
Logs directory: logs
Dataset: autism
Models: bilstm, stacked_lstm, gru, autoencoder, vae
======================================================================

Searching for bilstm checkpoints...
  ✓ Found checkpoint: logs/train/runs/2025-12-05_17-10-31/checkpoints/epoch_023.ckpt
    Validation accuracy: 0.8500

Searching for stacked_lstm checkpoints...
  ✓ Found checkpoint: logs/train/runs/2025-12-05_18-09-13/checkpoints/epoch_019.ckpt
    Validation accuracy: 0.8300

... (continues for all models)

✓ Found 5/5 model checkpoints.
✓ Generated ensemble configuration: configs/ensemble/autism_voting_auto.yaml
```

### 2. Run Ensemble Evaluation

Evaluate the ensemble on the test set:

```bash
python src/ensemble_eval.py ensemble=autism_voting_auto
```

**What happens**:
1. Loads all 5 models from checkpoints
2. Verifies preprocessing consistency
3. Computes weighted ensemble predictions
4. Evaluates on test set
5. Generates metrics and visualizations

**Expected runtime**: 2-5 minutes (depending on test set size and hardware)

### 3. View Results

Results are saved to `logs/ensemble_eval/runs/{timestamp}/ensemble_results/`:

```bash
# View comparison table
cat logs/ensemble_eval/runs/*/ensemble_results/model_comparison.csv

# View ensemble metrics
cat logs/ensemble_eval/runs/*/ensemble_results/ensemble_metrics.txt

# View plots
ls logs/ensemble_eval/runs/*/ensemble_results/plots/
```

**Expected output structure**:
```
ensemble_results/
├── model_comparison.csv           # ← Start here!
├── ensemble_metrics.txt
├── variance_analysis.txt
├── predictions.npz
├── confusion_matrix.csv
└── plots/
    ├── roc_comparison.png         # ← Publication-ready!
    ├── metric_comparison.png
    ├── variance_analysis.png
    └── confusion_matrix.png
```

## Common Scenarios

### Scenario 1: Missing Some Models

If you haven't trained all 5 models yet, you can run ensemble with available models:

```bash
# Discover only bilstm, gru, and autoencoder
python scripts/discover_ensemble_checkpoints.py \
    --dataset autism \
    --models bilstm gru autoencoder
```

The ensemble will work with any number of models ≥ 2.

### Scenario 2: Manual Configuration

If auto-discovery doesn't work, manually edit `configs/ensemble/autism_voting.yaml`:

1. Find your checkpoint paths:
   ```bash
   find logs/train/runs/ -name "*.ckpt" | grep -E "(bilstm|gru|lstm|autoencoder|vae)"
   ```

2. Edit config:
   ```yaml
   ensemble:
     models:
       - name: "bilstm"
         checkpoint_path: "logs/train/runs/2025-12-05_17-10-31/checkpoints/epoch_023.ckpt"
         val_accuracy: 0.85  # From training logs
   ```

3. Run ensemble:
   ```bash
   python src/ensemble_eval.py ensemble=autism_voting
   ```

### Scenario 3: Using CPU (No GPU)

```bash
python src/ensemble_eval.py ensemble=autism_voting_auto ensemble.device=cpu
```

### Scenario 4: Quick Test (Subset of Data)

For testing, use a small batch:

```bash
python src/ensemble_eval.py ensemble=autism_voting_auto data.batch_size=16
```

## Troubleshooting

### Issue 1: "No checkpoints found"

**Cause**: Models not trained yet or wrong logs directory.

**Fix**:
```bash
# Check if logs exist
ls -la logs/train/runs/

# Specify custom logs directory
python scripts/discover_ensemble_checkpoints.py \
    --logs-dir /path/to/logs \
    --dataset autism
```

### Issue 2: "Checkpoint not found"

**Cause**: Checkpoint path in config is invalid.

**Fix**:
```bash
# Verify checkpoint exists
ls -la logs/train/runs/*/checkpoints/*.ckpt

# Re-run discovery
python scripts/discover_ensemble_checkpoints.py --dataset autism
```

### Issue 3: "Preprocessing parameters inconsistent"

**Cause**: Models trained with different data configurations.

**Fix**: Ensure all models use same data config during training:
```bash
# Train all models with same data config
python src/train.py experiment=autism_bilstm data=autism
python src/train.py experiment=autism_gru data=autism
# etc.
```

### Issue 4: "CUDA out of memory"

**Cause**: All 5 models loaded on GPU.

**Fix**:
```bash
# Use CPU
python src/ensemble_eval.py ensemble=autism_voting_auto ensemble.device=cpu

# Or reduce batch size
python src/ensemble_eval.py ensemble=autism_voting_auto data.batch_size=8
```

## Next Steps

1. **Analyze Results**: Open `model_comparison.csv` and `roc_comparison.png`
2. **Compare Metrics**: Check if ensemble outperforms individual models
3. **Variance Analysis**: Review `variance_analysis.txt` for model agreement
4. **Publication**: Use plots in `plots/` folder (300 DPI, publication-ready)

## Advanced Usage

See [`docs/ensemble_usage.md`](./ensemble_usage.md) for:
- Custom weighting strategies
- Adding/removing models
- Cross-validation ensemble
- Mental health dataset
- API reference

## Getting Help

If you encounter issues:

1. Check error message carefully
2. Review troubleshooting section above
3. Verify all models are trained
4. Check configuration files
5. Run unit tests: `python tests/test_ensemble.py`

## Summary

```bash
# Three commands to run ensemble:
python scripts/discover_ensemble_checkpoints.py --dataset autism
python src/ensemble_eval.py ensemble=autism_voting_auto
cat logs/ensemble_eval/runs/*/ensemble_results/model_comparison.csv
```

**That's it!** 🎉
