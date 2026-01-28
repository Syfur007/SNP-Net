# Learning Rate Scheduler Configuration

This document explains the available learning rate schedulers and how to use them in SNP-Net.

## Overview

Learning rate schedulers adjust the learning rate during training to improve convergence and model performance. The framework now supports multiple scheduler types:

1. **CosineAnnealingLR** - Recommended for fixed-duration training
2. **CosineAnnealingWarmRestarts** - For exploration and escaping local minima
3. **ReduceLROnPlateau** - Default metric-based scheduler (adaptive)
4. **StepLR** - Simple step-based decay

---

## Available Schedulers

### 1. CosineAnnealingLR (RECOMMENDED for SNP-Net)

Reduces learning rate following a cosine curve from initial value to `eta_min` over `T_max` epochs.

**Best for:** Training with known max epochs (e.g., `trainer.max_epochs=100`)

**Config:** `configs/model/scheduler/cosine_annealing.yaml`

```yaml
_target_: torch.optim.lr_scheduler.CosineAnnealingLR
_partial_: true
T_max: 100          # Maximum number of epochs for cosine annealing
eta_min: 1.0e-6     # Minimum learning rate floor
```

**Usage:**
```bash
# Train with CosineAnnealingLR scheduler
python src/train.py model=bilstm model.scheduler=cosine_annealing trainer.max_epochs=100
```

**Advantages:**
- Smooth, predictable LR decay
- Better generalization than step-based decay
- No validation metric monitoring required
- Efficient computation

**When to use:**
- When you know the exact number of training epochs
- For faster training with stable convergence
- When you want smooth LR scheduling

---

### 2. CosineAnnealingWarmRestarts (SGDR)

Uses multiple restarts with increasing periods for periodic LR resets.

**Best for:** Exploration of solution space, longer training runs

**Config:** `configs/model/scheduler/cosine_annealing_warm_restarts.yaml`

```yaml
_target_: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
_partial_: true
T_0: 10             # Length of first restart period
T_mult: 2           # Period multiplier (period doubles each restart)
eta_min: 1.0e-6     # Minimum learning rate
```

**Usage:**
```bash
python src/train.py model=bilstm model.scheduler=cosine_annealing_warm_restarts trainer.max_epochs=200
```

**How it works:**
- Epoch 0-9: Cosine annealing with T_0=10
- Epoch 10-29: Cosine annealing with T_0=20 (T_0*T_mult)
- Epoch 30-69: Cosine annealing with T_0=40
- And so on...

**Advantages:**
- Multiple LR "resets" help escape local minima
- Can find better final solution
- Effective for longer training

**When to use:**
- When you have budget for longer training
- To improve model robustness
- For hyperparameter sensitivity analysis

---

### 3. ReduceLROnPlateau (DEFAULT)

Reduces LR when validation metric stops improving.

**Best for:** Training without knowing exact max epochs

**Config:** `configs/model/scheduler/reduce_lr_on_plateau.yaml` (currently default)

```yaml
_target_: torch.optim.lr_scheduler.ReduceLROnPlateau
_partial_: true
mode: min           # min for loss, max for accuracy/auroc
factor: 0.5         # Multiply LR by this factor
patience: 5         # Epochs with no improvement before reduction
min_lr: 1.0e-6      # Minimum learning rate
```

**Usage:** (Default - no changes needed)
```bash
python src/train.py model=bilstm  # Uses ReduceLROnPlateau by default
```

**Advantages:**
- Adaptive to actual convergence behavior
- Works regardless of max_epochs setting
- Reduces LR only when needed

**When to use:**
- When max_epochs is unknown/flexible
- For adaptive convergence
- When validation metric is reliable

---

### 4. StepLR

Reduces LR by gamma every `step_size` epochs.

**Best for:** Simple, predictable schedules

**Config:** `configs/model/scheduler/step_lr.yaml`

```yaml
_target_: torch.optim.lr_scheduler.StepLR
_partial_: true
step_size: 10       # Reduce LR every N epochs
gamma: 0.5          # Multiplicative factor
```

**Usage:**
```bash
python src/train.py model=bilstm model.scheduler=step_lr trainer.max_epochs=100
```

**Advantages:**
- Simple and interpretable
- Works well with known schedule
- Lower computation overhead

**When to use:**
- For quick experiments
- When you want simple LR schedule
- As a baseline for comparison

---

## Configuration Files

All scheduler configs are located in: `configs/model/scheduler/`

```
configs/model/scheduler/
├── cosine_annealing.yaml              # RECOMMENDED
├── cosine_annealing_warm_restarts.yaml
├── reduce_lr_on_plateau.yaml          # DEFAULT
└── step_lr.yaml
```

All model configs now support scheduler selection via Hydra defaults:

```yaml
# configs/model/lstm.yaml
defaults:
  - scheduler: reduce_lr_on_plateau  # Change this to switch schedulers
```

---

## Usage Examples

### Example 1: Train BiLSTM with Cosine Annealing

```bash
python src/train.py \
  model=bilstm \
  model.scheduler=cosine_annealing \
  trainer.max_epochs=100 \
  model.optimizer.lr=0.001
```

### Example 2: Compare schedulers

```bash
# Run with ReduceLROnPlateau (default)
python src/train.py model=transformer_cnn trainer.max_epochs=100

# Run with CosineAnnealingLR
python src/train.py model=transformer_cnn model.scheduler=cosine_annealing trainer.max_epochs=100

# Run with StepLR
python src/train.py model=transformer_cnn model.scheduler=step_lr trainer.max_epochs=100
```

### Example 3: Custom T_max for CosineAnnealingLR

```bash
python src/train.py \
  model=dense \
  model.scheduler=cosine_annealing \
  model.scheduler.T_max=150 \
  trainer.max_epochs=150
```

### Example 4: Experiment with warm restarts

```bash
python src/train.py \
  model=gru \
  model.scheduler=cosine_annealing_warm_restarts \
  model.scheduler.T_0=20 \
  model.scheduler.T_mult=2 \
  trainer.max_epochs=200
```

---

## Recommendations for SNP-Net Models

| Model | Recommended Scheduler | Reason | Notes |
|-------|----------------------|--------|-------|
| Dense | CosineAnnealingLR | Fast convergence | T_max=50-100 |
| LSTM | CosineAnnealingLR | Smooth decay | T_max=100-150 |
| BiLSTM | CosineAnnealingLR | Works well | T_max=100-150 |
| GRU | CosineAnnealingLR | Good performance | T_max=80-120 |
| Stacked LSTM | CosineAnnealingWarmRestarts | Explore solution space | T_0=10, T_mult=2 |
| Transformer-CNN | CosineAnnealingLR | Standard for transformers | T_max=100-200 |
| DPCformer | CosineAnnealingLR | Works well | T_max=100-200 |
| Autoencoder | ReduceLROnPlateau | Reconstruction loss varies | Default settings fine |
| VAE | ReduceLROnPlateau | KL divergence behavior | Default settings fine |
| WheatGP | CosineAnnealingLR | Complex model | T_max=150-200 |

---

## Implementation Details

### How Schedulers Are Used

1. **Config Definition**: Each scheduler is defined in YAML with Hydra `_target_` and parameters
2. **Factory Pattern**: Configs use `_partial_: true` to create partial functions
3. **LitModule Integration**: `configure_optimizers()` method handles scheduler instantiation
4. **Type Detection**: The code automatically detects metric-based vs epoch-based schedulers

### Code Example (in `src/models/module.py`):

```python
def configure_optimizers(self) -> Dict[str, Any]:
    optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
    if self.hparams.scheduler is not None:
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        
        # ReduceLROnPlateau needs metric monitoring
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler_config["monitor"] = "val/loss"
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
    return {"optimizer": optimizer}
```

---

## Advanced Usage

### Custom Scheduler Config

To create a custom scheduler config, add to `configs/model/scheduler/`:

```yaml
# configs/model/scheduler/exponential_lr.yaml
_target_: torch.optim.lr_scheduler.ExponentialLR
_partial_: true
gamma: 0.95  # Multiply LR by 0.95 each epoch
```

Then use:
```bash
python src/train.py model=bilstm model.scheduler=exponential_lr
```

### Combining with Hyperparameter Search

All schedulers work with Optuna hyperparameter search:

```bash
python src/train.py -m hparams_search=optuna model=bilstm model.scheduler=cosine_annealing
```

---

## Troubleshooting

### Issue: "KeyError: 'scheduler'"
**Solution**: Ensure defaults list is in your model config:
```yaml
defaults:
  - scheduler: cosine_annealing
```

### Issue: LR not changing
**For CosineAnnealingLR**: Check that `T_max` matches `trainer.max_epochs`
**For ReduceLROnPlateau**: Check that validation loss is being logged

### Issue: Training diverges
- **Reduce initial LR**: `model.optimizer.lr=0.0001`
- **Use ReduceLROnPlateau**: More conservative with LR reduction
- **Use smaller eta_min**: `model.scheduler.eta_min=1e-7`

---

## References

- PyTorch LR Schedulers: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- PyTorch Lightning Schedulers: https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.LightningModule.html#configure-optimizers
- CosineAnnealingLR Paper: https://arxiv.org/abs/1608.03983
- SGDR (Warm Restarts): https://arxiv.org/abs/1608.03983
