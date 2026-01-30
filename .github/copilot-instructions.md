# Copilot / AI Agent Quick Instructions ✅

Purpose: short, actionable orientation so an AI agent can be productive immediately in this repo.

## Big picture 🔧
- This project is a Lightning + Hydra template for training/evaluating models. Entry points:
  - `src/train.py` — training (supports k-fold via `data.num_folds` and Hydra multiruns)
  - `src/eval.py` — evaluation on a checkpoint (`ckpt_path` required)
- Config-driven design: almost every runtime object (datamodule, model, trainer, callbacks, loggers) is created via Hydra `DictConfig` + `hydra.utils.instantiate` using `_target_` keys. See `configs/` for canonical examples.
- `rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)` is used in entry points to add project root to `PYTHONPATH` and to set `PROJECT_ROOT` env var. The code is runnable without installing the package.

## Where to look first 📂 (high ROI files)
- `configs/` — primary Hydra configs (notably `train.yaml`, `eval.yaml`, `hydra/default.yaml`, `paths/default.yaml`).
- `src/train.py` & `src/eval.py` — show the runtime flow, logging, checkpoint behavior.
- `src/utils/` — helpers: `instantiators.py` (instantiate patterns), `utils.py` (extras, `task_wrapper`, `get_metric_value`), `logging_utils.py` (what gets logged).
- `src/models/components/` — concrete model definitions used by `_target_` configs.
- `Makefile`, `.pre-commit-config.yaml`, `pyproject.toml` — developer tooling and test config.
- `tests/` — examples of expected behavior and how features are exercised by CI.

## Key developer workflows & commands ▶️
- Local quick test: `make test` (runs `pytest -k "not slow"`). Full test suite: `make test-full` (`pytest`).
- Format & hooks: `make format` ⇒ runs pre-commit hooks (or `pre-commit run -a`).
- Run training: `python src/train.py` (Hydra overrides available: e.g. `trainer.max_epochs=20 model.optimizer.lr=1e-4`)
- Resume training / eval: `python src/train.py ckpt_path="/path/to.ckpt"` / `python src/eval.py ckpt_path="/path/to.ckpt"`
- Multirun / sweep: `python src/train.py -m data.batch_size=32,64 model.lr=0.001,0.0005` or `python src/train.py -m 'experiment=glob(*)'`
- Optuna sweep: defined under `configs/hparams_search/*` (note: Optuna sweeps are not failure-resistant).

## Conventions & patterns to respect 🧭
- Hydra config composition is the source of truth. Prefer changes in `configs/` over ad hoc CLI overrides for reproducibility.
- `callbacks` and `logger` configs are dictionaries where each value is a `DictConfig` with `_target_` to instantiate.
- `extras` config controls non-essential behavior (e.g., `ignore_warnings`, `enforce_tags`, `print_config`) and is applied at process startup via `extras(cfg)`.
- Metric extraction for HPO uses `optimized_metric` and `get_metric_value(metric_dict, metric_name)` — follow the metric naming conventions used in Lightning callbacks (often `avg_test/acc` vs `test/acc`).
- Output locations are Hydra-managed: `cfg.paths.output_dir` (maps to `hydra.runtime.output_dir`) and log location pattern is defined in `configs/hydra/default.yaml`.

## Integration points & external deps ⚙️
- Lightning loggers (W&B, Tensorboard, CSV, etc.) are configured through `configs/logger/*` and instantiated via `instantiate_loggers`.
- Optional integrations in codebase: `wandb` (closed in `task_wrapper` cleanup), `optuna` (via Hydra plugin), typical pytorch-lightning ecosystem.

## Testing advice for changes ✅
- Add unit tests under `tests/`. Use existing fixtures for data and datamodules.
- Use `pytest -k "not slow"` for quick local feedback; run full suite in CI.
- When changing config schemas, update or add a sample `configs/experiment/` file and a test that composes it.

## Quick examples to copy-paste 💡
- Run short training: `python src/train.py trainer.max_epochs=1 trainer.limit_train_batches=0.1 extras.print_config=true`
- Run k-fold CV: add `data.num_folds: 5` to config (or override on CLI). Training will run `train_kfold` automatically.
- Evaluate a checkpoint: `python src/eval.py ckpt_path="logs/.../checkpoints/epoch_002.ckpt"`

---
If anything here is unclear or you want more detail in a specific area (e.g., HPO setup, logger internals, or how tests mock dataloaders), tell me which section to expand and I will iterate. ✨