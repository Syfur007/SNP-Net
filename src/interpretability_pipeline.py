"""
Hydra-integrated interpretability analysis pipeline.

Orchestrates SHAP, IG, and LIME analyses across multiple models and datasets
with comprehensive cross-method comparison and robustness analysis.

Usage:
    python src/interpretability_pipeline.py checkpoint_dir=logs/train/runs/<experiment_name>_<date>
    python src/interpretability_pipeline.py --config-name=interpretability
"""

import argparse
import logging
import os
import sys
import subprocess
import shutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_checkpoint_dataset(checkpoint_path: str) -> Optional[str]:
    """Extract the training dataset name from a checkpoint's Hydra config.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., logs/train/runs/xxx/checkpoints/last.ckpt)
        
    Returns:
        Dataset name (e.g., 'GSE139294') or None if not found
    """
    import yaml
    
    ckpt_path = Path(checkpoint_path)
    run_dir = ckpt_path.parent.parent  # Go up from checkpoints/ to run directory
    config_file = run_dir / '.hydra' / 'config.yaml'
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract data_file path
        data_file = config.get('data', {}).get('data_file', '')
        if not data_file:
            return None
        
        # Extract dataset name from filename (e.g., 'data/Finalized_GSE139294.csv' -> 'GSE139294')
        filename = Path(data_file).stem  # e.g., 'Finalized_GSE139294'
        if 'GSE' in filename:
            # Extract GSE identifier
            parts = filename.split('GSE')
            if len(parts) > 1:
                return 'GSE' + parts[1]  # e.g., 'GSE139294'
        
        return None
    except Exception as e:
        logger.error(f"Error reading config for {checkpoint_path}: {e}")
        return None


def discover_checkpoints(mode: str = 'best', checkpoint_list: Optional[List[str]] = None) -> List[str]:
    """Discover checkpoint files.
    
    Args:
        mode: 'best' (latest best checkpoint), 'all' (all checkpoints), or 'custom'
        checkpoint_list: Explicit list of checkpoint paths (for custom mode)
        
    Returns:
        List of checkpoint file paths
    """
    if mode == 'custom' and checkpoint_list:
        return checkpoint_list
    
    log_dir = Path('logs/train/runs')
    if not log_dir.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return []
    
    checkpoints = []
    
    for run_dir in sorted(log_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        
        checkpoints_dir = run_dir / 'checkpoints'
        if checkpoints_dir.exists():
            if mode == 'best':
                # Try best.ckpt first, fall back to last.ckpt
                best_ckpt = checkpoints_dir / 'best.ckpt'
                if best_ckpt.exists():
                    checkpoints.append(str(best_ckpt))
                else:
                    # Fall back to last.ckpt
                    last_ckpt = checkpoints_dir / 'last.ckpt'
                    if last_ckpt.exists():
                        checkpoints.append(str(last_ckpt))
            elif mode == 'all':
                for ckpt in checkpoints_dir.glob('*.ckpt'):
                    checkpoints.append(str(ckpt))
    
    logger.info(f"Discovered {len(checkpoints)} checkpoints (mode={mode})")
    return checkpoints


def run_shap_analysis(
    checkpoint_path: str,
    data_file: str,
    output_dir: str,
    num_background: int = 50,
    num_test: int = 100,
    device: str = 'cpu',
) -> bool:
    """Run SHAP explainability analysis.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_file: Path to data CSV file
        output_dir: Output directory for results
        num_background: Number of background samples
        num_test: Number of test samples
        device: Computation device
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        result = subprocess.run([
            sys.executable, 'src/shap_explainability.py',
            '--checkpoint_path', checkpoint_path,
            '--data_file', data_file,
            '--output_dir', output_dir,
            '--num_background', str(num_background),
            '--num_test', str(num_test),
            '--device', device,
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"SHAP analysis failed: {result.stderr}")
            return False
        
        logger.info(f"✓ SHAP analysis completed for {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error running SHAP analysis: {e}")
        return False


def run_ig_analysis(
    checkpoint_path: str,
    data_file: str,
    output_dir: str,
    num_test: int = 100,
    baseline_type: str = 'mean',
    n_steps: int = 50,
    device: str = 'cpu',
) -> bool:
    """Run Integrated Gradients explainability analysis.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_file: Path to data CSV file
        output_dir: Output directory for results
        num_test: Number of test samples
        baseline_type: Type of baseline ('zero', 'mean', 'random')
        n_steps: Number of integration steps
        device: Computation device
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        result = subprocess.run([
            sys.executable, 'src/integrated_gradients_explainability.py',
            '--checkpoint_path', checkpoint_path,
            '--data_file', data_file,
            '--output_dir', output_dir,
            '--num_test', str(num_test),
            '--baseline', baseline_type,
            '--n_steps', str(n_steps),
            '--device', device,
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"IG analysis failed: {result.stderr}")
            return False
        
        logger.info(f"✓ IG analysis completed for {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error running IG analysis: {e}")
        return False


def run_lime_analysis(
    checkpoint_path: str,
    data_file: str,
    output_dir: str,
    num_test: int = 100,
    num_samples_per_test: int = 100,
    device: str = 'cpu',
) -> bool:
    """Run LIME explainability analysis.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_file: Path to data CSV file
        output_dir: Output directory for results
        num_test: Number of test samples
        num_samples_per_test: Number of perturbations per sample
        device: Computation device
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        result = subprocess.run([
            sys.executable, 'src/lime_explainability.py',
            '--checkpoint_path', checkpoint_path,
            '--data_file', data_file,
            '--output_dir', output_dir,
            '--num_test', str(num_test),
            '--num_samples_per_test', str(num_samples_per_test),
            '--device', device,
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"LIME analysis failed: {result.stderr}")
            return False
        
        logger.info(f"✓ LIME analysis completed for {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error running LIME analysis: {e}")
        return False


def run_analysis_pipeline(cfg: DictConfig) -> None:
    """Run the complete interpretability analysis pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("="*80)
    logger.info("STARTING INTERPRETABILITY ANALYSIS PIPELINE")
    logger.info("="*80)
    
    # Create output directories
    output_base = Path(cfg.analysis.output_base_dir)
    figure_dir = Path(cfg.analysis.figure_dir)
    data_dir = Path(cfg.analysis.data_dir)
    
    output_base.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_base}")
    
    # Determine device
    device = cfg.parallel.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Discover checkpoints
    checkpoint_selection = cfg.checkpoint_selection
    checkpoints = discover_checkpoints(
        mode=checkpoint_selection.mode,
        checkpoint_list=checkpoint_selection.get('checkpoints')
    )
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints to analyze")
    
    # Run analysis for each checkpoint with its training dataset only
    results = {
        'summary': [],
        'failed_analyses': []
    }
    
    for checkpoint_path in checkpoints:
        checkpoint_name = Path(checkpoint_path).parent.parent.name
        logger.info(f"\nProcessing checkpoint: {checkpoint_name}")
        
        # Determine which dataset this checkpoint was trained on
        trained_dataset = get_checkpoint_dataset(checkpoint_path)
        if not trained_dataset:
            logger.warning(f"  Could not determine training dataset for {checkpoint_name}, skipping...")
            continue
        
        # Find the matching dataset config
        dataset_config = None
        for ds_cfg in cfg.datasets:
            if ds_cfg.name == trained_dataset:
                dataset_config = ds_cfg
                break
        
        if not dataset_config:
            logger.warning(f"  Training dataset {trained_dataset} not in analysis config, skipping...")
            continue
        
        dataset_name = dataset_config.name
        data_file = dataset_config.data_file
        
        logger.info(f"  Analyzing with training dataset: {dataset_name}")
        
        # Create dataset-specific output directories
        dataset_output_base = output_base / checkpoint_name / dataset_name
        
        # Run SHAP analysis
        if cfg.methods.shap.enabled:
            shap_output = dataset_output_base / 'shap'
            shap_output.mkdir(parents=True, exist_ok=True)
            
            success = run_shap_analysis(
                checkpoint_path,
                data_file,
                str(shap_output),
                num_background=cfg.methods.shap.num_background,
                num_test=cfg.methods.shap.num_test,
                device=device,
            )
            
            if not success:
                results['failed_analyses'].append({
                    'checkpoint': checkpoint_name,
                    'dataset': dataset_name,
                    'method': 'SHAP'
                })
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Run IG analysis
        if cfg.methods.ig.enabled:
            ig_output = dataset_output_base / 'ig'
            ig_output.mkdir(parents=True, exist_ok=True)
            
            success = run_ig_analysis(
                checkpoint_path,
                data_file,
                str(ig_output),
                num_test=cfg.methods.ig.num_test,
                baseline_type=cfg.methods.ig.baseline,
                n_steps=cfg.methods.ig.n_steps,
                device=device,
            )
            
            if not success:
                results['failed_analyses'].append({
                    'checkpoint': checkpoint_name,
                    'dataset': dataset_name,
                    'method': 'IG'
                })
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Run LIME analysis
        if cfg.methods.lime.enabled:
            lime_output = dataset_output_base / 'lime'
            lime_output.mkdir(parents=True, exist_ok=True)
            
            success = run_lime_analysis(
                checkpoint_path,
                data_file,
                str(lime_output),
                num_test=cfg.methods.lime.num_test,
                num_samples_per_test=cfg.methods.lime.num_samples_per_test,
                device=device,
            )
            
            if not success:
                results['failed_analyses'].append({
                    'checkpoint': checkpoint_name,
                    'dataset': dataset_name,
                    'method': 'LIME'
                })
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results and summary
    logger.info("\n" + "="*80)
    logger.info("INTERPRETABILITY ANALYSIS PIPELINE COMPLETE")
    logger.info("="*80)
    
    # Save configuration used
    config_save_path = output_base / 'analysis_config.yaml'
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Save results summary
    results_json_path = output_base / 'analysis_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results summary saved to: {results_json_path}")
    
    if results['failed_analyses']:
        logger.warning(f"\n{len(results['failed_analyses'])} analyses failed:")
        for failed in results['failed_analyses']:
            logger.warning(f"  - {failed['checkpoint']} / {failed['dataset']} ({failed['method']})")
    
    logger.info(f"All results saved to: {output_base}")


@hydra.main(version_base="1.2", config_path="../configs/analysis", config_name="interpretability")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    run_analysis_pipeline(cfg)


if __name__ == '__main__':
    main()
