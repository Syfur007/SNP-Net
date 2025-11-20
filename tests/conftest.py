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
            # Use full training set in CI/tests to avoid fractional
            # limits resulting in <1 batch (Lightning raises MisconfigurationException).
            cfg.trainer.limit_train_batches = 1.0
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

    Uses a locally-created pytest.MonkeyPatch so the fixture can be session-scoped.
    """
    # Create a small deterministic CSV file that the DataModule expects.
    # Creating a real file is necessary so child processes (ddp_spawn) can see it.
    data_file = Path("data/snp_data.csv")
    if not data_file.exists():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        n_samples = 120
        n_features = 30

        # deterministic pseudo-random generation: use numpy with fixed seed
        rng = np.random.RandomState(12345)
        # create features with slight signal correlated with labels
        base = rng.randint(0, 3, size=(n_samples, n_features))
        # define labels as parity of sum of first feature column to make task learnable
        labels = (base.sum(axis=1) % 2).astype(int)

        # transpose to SNPs as rows, samples as columns and append label row
        full = np.vstack([base.T, labels.reshape(1, -1)])

        sample_names = [f"sample_{i}" for i in range(n_samples)]
        snp_names = [f"SNP_{i}" for i in range(n_features)]
        snp_names.append("label")

        with open(data_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + sample_names)
            for i, name in enumerate(snp_names):
                writer.writerow([name] + full[i].tolist())

    # set deterministic seeds for reproducibility across train/resume runs
    try:
        import torch

        torch.manual_seed(12345)
    except Exception:
        pass
    try:
        np.random.seed(12345)
    except Exception:
        pass

    yield None
