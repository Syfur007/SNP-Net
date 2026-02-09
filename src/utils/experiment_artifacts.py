import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def get_export_dir(cfg: DictConfig) -> Path:
    export_dir = Path(cfg.paths.output_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def get_env_context() -> Dict[str, Any]:
    return {
        "experiment_id": os.getenv("SNP_EXPERIMENT_ID"),
        "model_name": os.getenv("SNP_MODEL_NAME"),
        "seed": os.getenv("SNP_SEED"),
        "task_name": os.getenv("SNP_TASK_NAME"),
        "fold": os.getenv("SNP_FOLD"),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2, sort_keys=True)


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_serializable(row)) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_serializable(row.get(k)) for k in fieldnames})


def save_numpy(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savez(path, **arrays)


def export_config(cfg: DictConfig, export_dir: Path) -> None:
    ensure_dir(export_dir)
    OmegaConf.save(cfg, export_dir / "config.yaml")
    save_json(export_dir / "config.json", OmegaConf.to_container(cfg, resolve=True))


def export_run_manifest(export_dir: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **get_env_context(),
    }
    if extra:
        manifest.update(extra)
    save_json(export_dir / "run_manifest.json", manifest)
