# Hyperparameter Optimization (Optuna + Hydra)

This document consolidates hyperparameter optimization guidance for SNP-Net. It explains how the repository performs automatic hyperparameter search using Optuna (TPE sampler) integrated with Hydra, describes the configurable search spaces, and provides practical guidance and compute/resource estimates for running optimization jobs.

Contents
- Overview
- Quick start
- Configs available
- Recommended search spaces (by model)
- Interpreting metrics and results
- Compute and resource estimates
- Example commands
- Troubleshooting & tips

---

## Overview

SNP-Net uses Hydra’s Optuna Sweeper plugin to run hyperparameter optimization (multirun mode). Optuna's TPE sampler performs Bayesian optimization: it explores randomly for an initial set of trials, builds a surrogate model of the performance landscape, and proposes new parameter sets expected to improve the target metric.

The repository provides:
- A general-purpose Optuna config: `configs/hparams_search/optuna.yaml`
- Model-specific Optuna configs: `optuna_dense.yaml`, `optuna_lstm.yaml`, `optuna_transformer_cnn.yaml`
- Utilities in `src/train.py` that support single-run and k-fold training and return a metric value used by the sweeper.

The default optimized metric is `avg_test/acc` (average test accuracy across folds), which is recommended for small or noisy datasets.

---

## Quick start

Recommended (Dense, autism dataset, 5-fold CV):

```bash
python train.py -m \
  hparams_search=optuna_dense \
  experiment=autism_dense \
  data.num_folds=5
```

This will run the configured number of Optuna trials and return the best-performing hyperparameter set.

---

## Configs available

- `configs/hparams_search/optuna.yaml` — general-purpose; optimizes LR, weight decay, dropout, batch size, scheduler parameters.
- `configs/hparams_search/optuna_dense.yaml` — tuned search space for dense networks (includes architecture choices).
- `configs/hparams_search/optuna_lstm.yaml` — tuned for LSTM/BiLSTM models (hidden size, layers, window size).
- `configs/hparams_search/optuna_transformer_cnn.yaml` — tuned for transformer–CNN hybrids (d_model, nhead, depth).

Each config is a Hydra multirun sweeper that defines `n_trials`, `n_startup_trials`, `sampler`, `n_jobs`, and `params` (the search space).

---

## Recommended search spaces (interpretation)

Core hyperparameters (apply to all models):
- `model.optimizer.lr` (learning rate): continuous. Single most important hyperparameter — controls step size in optimization.
- `model.optimizer.weight_decay`: continuous. L2 regularization strength, controls overfitting.
- `model.net.dropout`: continuous 0.15–0.45. Regularization; prevents co-adaptation.
- `data.batch_size`: discrete choices (16, 32, 48, 64). Trade-off between gradient noise and stability.
- `model.scheduler.patience` and `model.scheduler.factor`: scheduler behavior for LR reduction.

Model-specific suggestions:
- Dense: search `model.net.hidden_sizes` (several fixed layer-size tuples). Larger nets increase capacity but risk overfitting.
- LSTM: `model.net.hidden_size`, `model.net.num_layers`, `model.net.window_size`. Window size determines how many SNPs form a sequence; larger windows capture long-range interactions but cost more compute.
- Transformer-CNN: `model.net.d_model`, `model.net.nhead`, `model.net.num_transformer_layers`. Transformers need smaller learning rates and more compute; nhead controls diversity of attention patterns.

---

## Interpreting metrics and results

Choose the optimization metric that matches your goals:
- `avg_test/acc`: average accuracy across folds — recommended for robust evaluation.
- `test/auroc`: better for imbalanced datasets.
- `test/f1`: balances precision and recall for skewed datasets.

Optuna returns the best trial summary with the parameter set and the objective value. Use that set to create a new experiment config and train a final model.

---

## Compute and resource estimates (practical guidance)

Estimates depend on dataset size, model architecture, whether training uses k-fold CV, and hardware. The numbers below are conservative, approximate, and intended to help plan resources.

Baseline assumptions for estimates below:
- Dataset: moderate (FinalizedAutismData.csv / FinalizedMentalData.csv). If your dataset is large (10k+ samples) expect much longer training times.
- Single GPU: NVIDIA RTX 3090 / A5000 / A100 class (>=24 GB VRAM) is ideal for transformer models; smaller GPUs (8-12 GB) work for dense/LSTM for modest batch sizes.

Per-trial time (single training run per trial):
- Dense network (single train/val/test split): ~2–10 minutes per trial on a single modern GPU; CPU-only runs are 3–10× slower.
- LSTM/BiLSTM (sequence models): ~5–20 minutes per trial on GPU.
- Transformer-CNN hybrid: ~20–90 minutes per trial on a single GPU depending on `num_snps`, `d_model`, and batch size.

K-fold multipliers:
- K-fold = 5 multiplies trial time roughly by 5 (each trial runs training for each fold).
- K-fold increases robustness but proportionally increases compute cost.

Full optimization job (examples):
- Quick search: 20 trials, Dense, single split, GPU — estimated 40–200 minutes (0.7–3.5 hours).
- Standard search: 40 trials, Dense, 5-fold CV, GPU — estimated 40 trials × per-trial-time × 5 ≈ (2–10 min × 40 × 5) = 6.7–33.3 hours. In practice, with early stopping and smaller epochs, times are usually in the 4–10 hour range for moderate datasets.
- Transformer search: 50 trials, Transformer-CNN, 5-fold CV, GPU — can take days on a single GPU. Consider reducing `n_trials`, running fewer folds, or using multiple GPUs / distributed jobs.

Resource guidance & recommendations:
- Dense and LSTM: a single GPU with 8–16 GB VRAM is generally sufficient for moderate batch sizes (16–64). Use 4–8 CPU cores for data loading (num_workers).
- Transformer-CNN: prefer GPUs with >=24 GB VRAM for larger `num_snps` (1000–2000) and larger batch sizes. If unavailable, reduce `num_snps` and batch size.
- Parallel trials (`n_jobs` > 1): only set if you have multiple GPUs or machines. Each parallel job consumes full training resources.
- Disk/IO: store Optuna DB (SQLite) on fast NVMe if possible when `storage` is enabled.
- Memory: ensure sufficient CPU RAM for data loading (~4–8 GB per worker as a rough guide for large datasets).

Tips to limit runtime:
- Reduce `n_trials` to 20–30 for exploratory runs.
- Use single split before committing to k-fold.
- Use `trainer.max_epochs` low (e.g., 10) during search and re-train final model with higher `max_epochs`.

---

## Example commands

Run a standard optimization for dense model with persistence (SQLite) and 5-fold CV:

```bash
python train.py -m \
  hparams_search=optuna_dense \
  experiment=autism_dense \
  data.num_folds=5 \
  hydra.sweeper.n_trials=40 \
  hydra.sweeper.storage="sqlite:///optuna_autism.db" \
  hydra.sweeper.study_name="autism_dense_kfold"
```

Run a transformer search with fewer trials to limit time:

```bash
python train.py -m \
  hparams_search=optuna_transformer_cnn \
  experiment=autism_transformer_cnn \
  hydra.sweeper.n_trials=20 \
  data.num_folds=3
```

Resume a stored search and increase trials:

```bash
python train.py -m \
  hparams_search=optuna_dense \
  experiment=autism_dense \
  hydra.sweeper.storage="sqlite:///optuna_autism.db" \
  hydra.sweeper.study_name="autism_dense_kfold" \
  hydra.sweeper.n_trials=80
```

---

## Troubleshooting & tips

- If trials fail with OOM, reduce batch size or `num_snps` (for transformer). Use smaller `d_model` or fewer `nhead`.
- If Optuna proposals repeat or don't improve, increase `n_startup_trials` to explore more randomly before switching to the TPE sampler.
- To debug a failing trial, run the same configuration manually by instantiating the best-trial parameters in a standard Hydra run (not multirun).
- Save results with `hydra.sweeper.storage="sqlite:///optuna.db"` to persist history and resume later.

---

## After optimization

1. Extract best trial parameters from Optuna output.
2. Create an `experiment` config file under `configs/experiment/` and set those parameter values.
3. Train a final model with the chosen config (larger `max_epochs`, optional full training data, and final test evaluation).

---

## References

- Optuna: https://optuna.readthedocs.io/
- Hydra Optuna Sweeper: https://hydra.cc/docs/plugins/optuna_sweeper/

---

(Previously the project included several separate hyperparameter docs; they have been consolidated into this single file.)
