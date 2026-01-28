"""Ensemble evaluation script for SNP classification.

This script evaluates a weighted soft-voting ensemble of pre-trained models
on the test set, ensuring identical preprocessing as during training.
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path

import hydra
import rootutils
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
import numpy as np

# Patch torch.load to disable weights_only check for PyTorch 2.6+ compatibility
_original_torch_load = torch.load

def _patched_torch_load(path, *args, **kwargs):
    """Patched torch.load that disables weights_only checks for custom classes."""
    kwargs['weights_only'] = False
    return _original_torch_load(path, *args, **kwargs)

torch.load = _patched_torch_load

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, register_checkpoint_safe_globals, task_wrapper
from src.ensemble import WeightedSoftVotingEnsemble, EnsembleEvaluator
from src.ensemble.preprocessing import (
    load_preprocessing_from_checkpoint,
    verify_preprocessing_consistency,
    get_preprocessing_summary,
    apply_preprocessing,
)
from src.ensemble.plotting import create_all_plots

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate_ensemble(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate weighted soft-voting ensemble on test set.
    
    Args:
        cfg: DictConfig configuration composed by Hydra.
        
    Returns:
        Tuple of (metric_dict, object_dict) with evaluation results.
    """
    # Register safe globals for checkpoint loading (PyTorch 2.6+)
    register_checkpoint_safe_globals()
    
    # Validate ensemble configuration
    if not cfg.get("ensemble"):
        raise ValueError("Ensemble configuration is required. Please specify 'ensemble' in config.")
    
    ensemble_cfg = cfg.ensemble
    
    log.info("=" * 70)
    log.info("ENSEMBLE EVALUATION")
    log.info("=" * 70)
    
    # Extract model configurations
    model_configs = ensemble_cfg.models
    if not model_configs:
        raise ValueError("No models specified in ensemble configuration.")
    
    log.info(f"Ensemble contains {len(model_configs)} models:")
    for model_cfg in model_configs:
        log.info(f"  - {model_cfg.name}: {model_cfg.checkpoint_path}")
    
    # Verify all checkpoints exist
    checkpoint_paths = [model_cfg.checkpoint_path for model_cfg in model_configs]
    for ckpt_path in checkpoint_paths:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Verify preprocessing consistency across checkpoints
    log.info("\nVerifying preprocessing consistency across checkpoints...")
    is_consistent, message = verify_preprocessing_consistency(checkpoint_paths)
    
    if not is_consistent:
        log.warning(f"Preprocessing inconsistency detected: {message}")
        if not ensemble_cfg.get("allow_inconsistent_preprocessing", False):
            raise ValueError(
                "Preprocessing parameters are inconsistent across checkpoints. "
                "Set 'ensemble.allow_inconsistent_preprocessing=true' to override."
            )
    else:
        log.info(f"✓ {message}")
    
    # Load preprocessing parameters from reference checkpoint
    reference_checkpoint = checkpoint_paths[0]
    log.info(f"\nLoading preprocessing parameters from reference checkpoint:")
    log.info(f"  {reference_checkpoint}")
    
    preprocessing_params = load_preprocessing_from_checkpoint(reference_checkpoint)
    
    log.info("\n" + get_preprocessing_summary(preprocessing_params))
    
    # Instantiate datamodule
    log.info(f"\nInstantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Setup datamodule to prepare datasets
    datamodule.setup(stage="test")
    
    # Get test dataloader
    test_dataloader = datamodule.test_dataloader()
    log.info(f"Test set size: {len(test_dataloader.dataset)} samples")
    
    # Create ensemble
    log.info("\nCreating weighted soft-voting ensemble...")
    
    checkpoint_dict = {
        model_cfg.name: model_cfg.checkpoint_path 
        for model_cfg in model_configs
    }
    
    validation_accuracies = {
        model_cfg.name: model_cfg.val_accuracy 
        for model_cfg in model_configs
    }
    
    device = ensemble_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg.data.get("num_classes", 2)
    
    ensemble = WeightedSoftVotingEnsemble(
        checkpoint_paths=checkpoint_dict,
        validation_accuracies=validation_accuracies,
        device=device,
        num_classes=num_classes,
    )
    
    log.info("\n" + str(ensemble))
    
    # Create evaluator
    log.info("\nInitializing ensemble evaluator...")
    task = "binary" if num_classes == 2 else "multiclass"
    
    evaluator = EnsembleEvaluator(
        ensemble=ensemble,
        task=task,
        num_classes=num_classes,
    )
    
    # Evaluate ensemble
    log.info("\nEvaluating ensemble on test set...")
    results = evaluator.evaluate(
        dataloader=test_dataloader,
        save_predictions=ensemble_cfg.get("save_predictions", True),
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    output_dir = Path(cfg.paths.output_dir) / "ensemble_results"
    log.info(f"\nSaving results to: {output_dir}")
    evaluator.save_results(output_dir)
    
    # Generate plots
    if ensemble_cfg.get("generate_plots", True):
        log.info("\nGenerating visualization plots...")
        plots_dir = output_dir / "plots"
        
        try:
            create_all_plots(
                results=results,
                model_names=ensemble.model_names,
                output_dir=plots_dir,
            )
        except Exception as e:
            log.warning(f"Failed to generate some plots: {str(e)}")
    
    # Prepare return values
    metric_dict = {
        "ensemble/accuracy": results["ensemble"]["accuracy"],
        "ensemble/precision": results["ensemble"]["precision"],
        "ensemble/recall": results["ensemble"]["recall"],
        "ensemble/f1": results["ensemble"]["f1"],
        "ensemble/auroc": results["ensemble"]["auroc"],
    }
    
    # Add individual model metrics
    for model_name, metrics in results["individual"].items():
        for metric_name, value in metrics.items():
            if metric_name != "confusion_matrix":
                metric_dict[f"{model_name}/{metric_name}"] = value
    
    # Add variance analysis
    variance = results["variance_analysis"]
    metric_dict["variance/mean_accuracy"] = variance["mean_accuracy"]
    metric_dict["variance/std_accuracy"] = variance["std_accuracy"]
    metric_dict["variance/mean_agreement"] = variance["mean_agreement"]
    
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "ensemble": ensemble,
        "evaluator": evaluator,
        "results": results,
    }
    
    log.info("\n" + "=" * 70)
    log.info("ENSEMBLE EVALUATION COMPLETE!")
    log.info("=" * 70)
    
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="ensemble_eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for ensemble evaluation.
    
    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # Apply extra utilities
    extras(cfg)
    
    # Evaluate ensemble
    evaluate_ensemble(cfg)


if __name__ == "__main__":
    main()
