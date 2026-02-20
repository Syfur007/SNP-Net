from __future__ import annotations

import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.utils.experiment_artifacts import (
    export_run_manifest,
    get_env_context,
    save_csv,
    save_json,
    save_jsonl,
    save_npz,
)


def _safe_item(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _compute_mcc_from_confusion(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    n = cm.sum()
    c = np.trace(cm)
    s = np.sum(p_sum * t_sum)
    numerator = c * n - s
    denominator = math.sqrt((n ** 2 - (p_sum ** 2).sum()) * (n ** 2 - (t_sum ** 2).sum()))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _compute_balanced_accuracy(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.diag(cm) / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
    return float(np.mean(recalls))


def _memory_stats() -> Dict[str, Any]:
    stats = {}
    if torch.cuda.is_available():
        stats["gpu/memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats["gpu/max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        stats["xpu/memory_allocated_mb"] = torch.xpu.memory_allocated() / (1024 ** 2)
        stats["xpu/max_memory_allocated_mb"] = torch.xpu.max_memory_allocated() / (1024 ** 2)
    try:
        import psutil

        process = psutil.Process(os.getpid())
        stats["cpu/rss_mb"] = process.memory_info().rss / (1024 ** 2)
    except Exception:
        pass
    return stats


def _collect_callback_metrics(trainer: Trainer) -> Dict[str, Any]:
    metrics = {}
    for key, value in trainer.callback_metrics.items():
        try:
            metrics[key] = _safe_item(value)
        except Exception:
            continue
    return metrics


@dataclass
class RobustnessConfig:
    noise_std: Iterable[float] = (0.0, 0.01, 0.05)
    feature_dropout: Iterable[float] = (0.0, 0.1, 0.25)
    subset_fractions: Iterable[float] = (1.0, 0.75, 0.5)


class ExperimentLoggerCallback(Callback):
    def __init__(
        self,
        export_dir: str,
        experiment_id: Optional[str] = None,
        model_name: Optional[str] = None,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        calibration_bins: int = 10,
        threshold_steps: int = 101,
        robustness: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.task_name = task_name
        self.seed = seed
        self.calibration_bins = calibration_bins
        self.threshold_steps = threshold_steps
        self.robustness = RobustnessConfig(**robustness) if robustness else RobustnessConfig()

        self._epoch_rows: List[Dict[str, Any]] = []
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._train_start_time = None
        self._epoch_start_time = None
        self._run_dir: Optional[Path] = None

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._train_start_time = time.time()
        env_ctx = get_env_context()
        fold = env_ctx.get("fold")
        seed = env_ctx.get("seed")
        run_dir = self.export_dir
        if fold is not None:
            run_dir = run_dir / f"fold_{fold}"
        if seed is not None:
            run_dir = run_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        export_run_manifest(
            run_dir,
            extra={
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "lightning_version": getattr(trainer, "__version__", None),
                "device": str(pl_module.device),
                "export_dir": str(run_dir),
                "model_params_total": total_params,
                "model_params_trainable": trainable_params,
                **env_ctx,
            },
        )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._epoch_start_time = time.time()
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0

    def on_after_backward(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self._grad_norm_sum += total_norm
        self._grad_norm_count += 1

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        metrics = _collect_callback_metrics(trainer)
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        lr = None
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get("lr")

        overfit_gap = None
        if "train/loss" in metrics and "val/loss" in metrics:
            try:
                overfit_gap = metrics["val/loss"] - metrics["train/loss"]
            except Exception:
                overfit_gap = None

        epoch_time = None
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time

        grad_norm = None
        if self._grad_norm_count > 0:
            grad_norm = self._grad_norm_sum / self._grad_norm_count

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": lr,
            "grad_norm": grad_norm,
            "epoch_time_sec": epoch_time,
            "overfitting_gap": overfit_gap,
            **metrics,
            **_memory_stats(),
            **get_env_context(),
        }
        self._epoch_rows.append(row)

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if self._run_dir is None:
            return

        if self._epoch_rows:
            save_jsonl(self._run_dir / "metrics_epoch.jsonl", self._epoch_rows)
            save_csv(self._run_dir / "metrics_epoch.csv", self._epoch_rows)

        summary = {
            "training_time_sec": (time.time() - self._train_start_time) if self._train_start_time else None,
            "early_stopped": trainer.should_stop,
            **get_env_context(),
        }

        early_stopping_cb = None
        for cb in trainer.callbacks:
            if cb.__class__.__name__.lower().startswith("earlystopping"):
                early_stopping_cb = cb
                break

        if early_stopping_cb is not None:
            summary.update(
                {
                    "early_stopping/patience": getattr(early_stopping_cb, "patience", None),
                    "early_stopping/wait_count": getattr(early_stopping_cb, "wait_count", None),
                    "early_stopping/stopped_epoch": getattr(early_stopping_cb, "stopped_epoch", None),
                }
            )

        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is not None:
            summary.update(
                {
                    "best_model_path": getattr(ckpt_cb, "best_model_path", None),
                    "best_model_score": _safe_item(getattr(ckpt_cb, "best_model_score", None)),
                }
            )

            best_path = getattr(ckpt_cb, "best_model_path", "")
            if best_path:
                try:
                    import re

                    match = re.search(r"epoch_(\d+)", best_path)
                    if match:
                        summary["best_epoch"] = int(match.group(1))
                except Exception:
                    pass

        save_json(self._run_dir / "training_summary.json", summary)

    @rank_zero_only
    def on_test_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if self._run_dir is None:
            return
        metrics = _collect_callback_metrics(trainer)
        save_json(self._run_dir / "test_metrics.json", {**metrics, **get_env_context()})

        probs = getattr(pl_module, "_test_probs", None)
        targets = getattr(pl_module, "_test_targets", None)
        if probs is not None and targets is not None:
            probs_np = _tensor_to_numpy(torch.cat(probs))
            targets_np = _tensor_to_numpy(torch.cat(targets))

            save_npz(self._run_dir / "test_predictions.npz", probs=probs_np, targets=targets_np)

            self._export_curves_and_calibration(probs_np, targets_np)
            self._export_threshold_sensitivity(probs_np, targets_np)
            self._export_robustness(trainer, pl_module)
            self._export_latent_representations(trainer, pl_module)
            self._export_attention_weights(trainer, pl_module)

        # Confusion matrices
        for stage_name, cm_attr in [
            ("train", "train_confusion_matrix"),
            ("val", "val_confusion_matrix"),
            ("test", "test_confusion_matrix"),
        ]:
            cm_metric = getattr(pl_module, cm_attr, None)
            if cm_metric is None:
                continue
            cm = cm_metric.compute().detach().cpu().numpy()
            save_npz(self._run_dir / f"{stage_name}_confusion_matrix.npz", cm=cm)
            save_json(
                self._run_dir / f"{stage_name}_confusion_matrix.json",
                {
                    "balanced_accuracy": _compute_balanced_accuracy(cm),
                    "mcc": _compute_mcc_from_confusion(cm),
                },
            )

    def _export_curves_and_calibration(self, probs: np.ndarray, targets: np.ndarray) -> None:
        if self._run_dir is None:
            return
        if probs.ndim == 2 and probs.shape[1] > 1:
            positive_probs = probs[:, 1]
        else:
            positive_probs = probs

        try:
            from sklearn.metrics import roc_curve, precision_recall_curve, auc

            fpr, tpr, roc_thresholds = roc_curve(targets, positive_probs)
            precision, recall, pr_thresholds = precision_recall_curve(targets, positive_probs)
            roc_auc = auc(fpr, tpr)
            pr_auc = auc(recall, precision)

            save_npz(
                self._run_dir / "test_curves.npz",
                fpr=fpr,
                tpr=tpr,
                roc_thresholds=roc_thresholds,
                precision=precision,
                recall=recall,
                pr_thresholds=pr_thresholds,
            )
            save_json(
                self._run_dir / "test_curve_metrics.json",
                {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
            )
        except Exception:
            pass

        # Calibration (ECE)
        bins = np.linspace(0.0, 1.0, self.calibration_bins + 1)
        bin_ids = np.digitize(positive_probs, bins) - 1
        bin_acc = []
        bin_conf = []
        bin_count = []
        for b in range(self.calibration_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                bin_acc.append(0.0)
                bin_conf.append(0.0)
                bin_count.append(0)
                continue
            bin_acc.append(float(targets[mask].mean()))
            bin_conf.append(float(positive_probs[mask].mean()))
            bin_count.append(int(mask.sum()))
        ece = sum(
            (bin_count[i] / len(targets)) * abs(bin_acc[i] - bin_conf[i])
            for i in range(self.calibration_bins)
        )
        save_json(
            self._run_dir / "test_calibration.json",
            {
                "ece": float(ece),
                "bins": bins.tolist(),
                "bin_acc": bin_acc,
                "bin_conf": bin_conf,
                "bin_count": bin_count,
            },
        )

    def _export_threshold_sensitivity(self, probs: np.ndarray, targets: np.ndarray) -> None:
        if self._run_dir is None:
            return
        if probs.ndim == 2 and probs.shape[1] > 1:
            positive_probs = probs[:, 1]
        else:
            positive_probs = probs

        thresholds = np.linspace(0.0, 1.0, self.threshold_steps)
        rows = []
        for t in thresholds:
            preds = (positive_probs >= t).astype(int)
            tp = ((preds == 1) & (targets == 1)).sum()
            tn = ((preds == 0) & (targets == 0)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            acc = (tp + tn) / max(1, len(targets))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            rows.append(
                {
                    "threshold": float(t),
                    "accuracy": float(acc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
            )
        save_csv(self._run_dir / "threshold_sensitivity.csv", rows)

    def _export_robustness(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if self._run_dir is None:
            return
        if trainer.datamodule is None:
            return
        dataloader = trainer.datamodule.test_dataloader()
        if dataloader is None:
            return

        pl_module.eval()
        rows = []
        device = pl_module.device

        def _evaluate_with_transform(transform_fn, tag: str) -> None:
            all_probs = []
            all_targets = []
            with torch.no_grad():
                for batch in dataloader:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    x = transform_fn(x)
                    logits = pl_module(x)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.detach().cpu())
                    all_targets.append(y.detach().cpu())
            probs_np = torch.cat(all_probs).numpy()
            targets_np = torch.cat(all_targets).numpy()
            if probs_np.ndim == 2 and probs_np.shape[1] > 1:
                positive_probs = probs_np[:, 1]
            else:
                positive_probs = probs_np
            preds = (positive_probs >= 0.5).astype(int)
            acc = float((preds == targets_np).mean())
            rows.append({"scenario": tag, "accuracy": acc})

        for std in self.robustness.noise_std:
            _evaluate_with_transform(lambda x, s=std: x + torch.randn_like(x) * s, f"noise_std_{std}")

        for rate in self.robustness.feature_dropout:
            def dropout_fn(x, r=rate):
                if r <= 0:
                    return x
                mask = torch.rand_like(x) > r
                return x * mask

            _evaluate_with_transform(dropout_fn, f"feature_dropout_{rate}")

        for frac in self.robustness.subset_fractions:
            def subset_fn(x, f=frac):
                if f >= 1.0:
                    return x
                k = max(1, int(x.size(1) * f))
                mask = torch.zeros_like(x)
                mask[:, :k] = 1.0
                return x * mask

            _evaluate_with_transform(subset_fn, f"subset_fraction_{frac}")

        if rows:
            save_csv(self._run_dir / "robustness_metrics.csv", rows)

    def _export_latent_representations(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if self._run_dir is None or trainer.datamodule is None:
            return

        net = getattr(pl_module, "net", None)
        if net is None:
            return

        if hasattr(net, "get_latent_representation"):
            latent_fn = lambda x: net.get_latent_representation(x, use_mean=True)
        elif hasattr(net, "encode"):
            latent_fn = lambda x: net.encode(x)
        elif hasattr(net, "encoder") and net.encoder is not None:
            latent_fn = lambda x: net.encoder(x)
        else:
            return

        dataloader = trainer.datamodule.test_dataloader()
        if dataloader is None:
            return

        latents = []
        targets = []
        device = pl_module.device
        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                z = latent_fn(x)
                latents.append(z.detach().cpu())
                targets.append(y.detach().cpu())

        if latents:
            save_npz(
                self._run_dir / "latent_representations.npz",
                latents=torch.cat(latents).numpy(),
                targets=torch.cat(targets).numpy(),
            )

    def _export_attention_weights(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if self._run_dir is None or trainer.datamodule is None:
            return

        net = getattr(pl_module, "net", None)
        if net is None:
            return

        dataloader = trainer.datamodule.test_dataloader()
        if dataloader is None:
            return

        # Run a single batch to capture attention weights if available
        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = x.to(pl_module.device)
                _ = pl_module(x)
                break

        attn = getattr(net, "last_attention_weights", None)
        if attn is None:
            return

        try:
            if isinstance(attn, list):
                arrays = {f"layer_{i}": a.detach().cpu().numpy() for i, a in enumerate(attn) if a is not None}
                if arrays:
                    save_npz(self._run_dir / "attention_weights.npz", **arrays)
            else:
                save_npz(self._run_dir / "attention_weights.npz", attn=attn.detach().cpu().numpy())
        except Exception:
            pass
