from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    patch_lightning_xpu_parse_devices,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path
    patch_lightning_xpu_parse_devices()

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    # Instantiate a model object, then try to load weights from checkpoint using
    # the class' `load_from_checkpoint` to avoid state_dict loading errors.
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    try:
        # Attempt to load a model instance directly from checkpoint. This will
        # construct the model using saved hyperparameters and restore weights.
        model_cls = model.__class__
        log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
        loaded = model_cls.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
        model = loaded
    except Exception as e:
        log.info(f"load_from_checkpoint failed ({e}), falling back to instantiated model and letting Trainer handle ckpt")

    # Load preprocessing parameters from checkpoint to ensure test set is processed
    # using the same statistics that were computed from the training set
    log.info(f"Loading preprocessing parameters from checkpoint: {cfg.ckpt_path}")
    try:
        import torch
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        datamodule_state = checkpoint.get("datamodule", checkpoint.get("DataModule", {}))
        
        if datamodule_state:
            log.info("Restoring preprocessing parameters from checkpoint...")
            datamodule.load_state_dict(datamodule_state)
            log.info("✓ Preprocessing parameters loaded (mean, std, feature selection indices)")
        else:
            log.warning("No datamodule state found in checkpoint. Preprocessing will be recomputed from test data.")
    except Exception as e:
        log.warning(f"Could not load preprocessing from checkpoint ({e}). Preprocessing will be recomputed from test data.")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
