"""This file prepares config fixtures for other tests."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            # Use full validation/test sets in CI tests to avoid fractional
            # limits resulting in <1 batch (Lightning raises MisconfigurationException).
            cfg.trainer.limit_val_batches = 1.0
            cfg.trainer.limit_test_batches = 1.0
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            # Ensure evaluation uses at least one full batch during tests
            cfg.trainer.limit_test_batches = 1.0
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="session", autouse=True)
def create_dummy_snp_data() -> None:
    """Autouse fixture to monkeypatch DataModule data loading for tests.

    Instead of creating files on disk, we replace `DataModule.prepare_data`
    with a no-op and `DataModule._load_data` with a function that returns
    small synthetic tensors. This avoids touching the filesystem in CI.
    """
    # Delay import until fixture time
    import torch
    from src.data.datamodule import DataModule

    def _fake_prepare_data(self):
        # do nothing (avoid FileNotFoundError in prepare_data)
        return None

    def _fake_load_data(self):
        # Return synthetic data: (samples x features) and labels (samples,)
        n_samples = 100
        n_features = 50
        data = torch.randint(0, 3, (n_samples, n_features), dtype=torch.float32)
        labels = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        return data, labels

    # Apply monkeypatch
    try:
        import pytest as _pytest

        # Use pytest's monkeypatch fixture if available in this context
        # since we're inside a fixture, request the built-in monkeypatch via import
        from _pytest.monkeypatch import MonkeyPatch

        mp = MonkeyPatch()
        mp.setattr(DataModule, "prepare_data", _fake_prepare_data, raising=False)
        mp.setattr(DataModule, "_load_data", _fake_load_data, raising=False)

        # register finalizer to undo patches at session end
        _pytest.fixture().request = None
    except Exception:
        # Fallback: set attributes directly
        DataModule.prepare_data = _fake_prepare_data
        DataModule._load_data = _fake_load_data
