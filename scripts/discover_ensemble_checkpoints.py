"""Helper script to discover trained model checkpoints and create ensemble configuration.

This script scans the logs directory for trained model checkpoints and generates
a properly configured ensemble YAML file with actual checkpoint paths and validation accuracies.
"""

import argparse
from pathlib import Path
import torch
import yaml
from typing import Dict, List, Optional
import sys

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def extract_val_accuracy_from_checkpoint(checkpoint_path: Path) -> Optional[float]:
    """Extract validation accuracy from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Validation accuracy if found, None otherwise.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try different possible keys
        callbacks_state = checkpoint.get('callbacks', {})
        
        # Look for ModelCheckpoint callback state (prioritize this)
        for callback_name, callback_state in callbacks_state.items():
            if not isinstance(callback_state, dict):
                continue
            
            # Check if this is ModelCheckpoint by looking for monitor field
            monitor = callback_state.get('monitor', '')
            
            # Only extract score from callbacks monitoring accuracy
            if 'acc' in monitor.lower():
                if 'best_model_score' in callback_state:
                    score = callback_state['best_model_score']
                    if hasattr(score, 'item'):
                        return score.item()
                    return float(score)
                if 'best_score' in callback_state:
                    score = callback_state['best_score']
                    if hasattr(score, 'item'):
                        return score.item()
                    return float(score)
        
        # Fallback: look in hyper_parameters
        hyper_params = checkpoint.get('hyper_parameters', {})
        if 'val_acc' in hyper_params:
            return hyper_params['val_acc']
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not extract validation accuracy from {checkpoint_path}: {e}")
        return None


def find_best_checkpoint(run_dir: Path, monitor_metric: str = "val/acc") -> Optional[Path]:
    """Find the best checkpoint in a run directory.
    
    Args:
        run_dir: Path to the run directory.
        monitor_metric: Metric used for checkpoint selection.
        
    Returns:
        Path to best checkpoint if found, None otherwise.
    """
    checkpoint_dir = run_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoints
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    
    if not checkpoints:
        return None
    
    # Try to find checkpoint with best in name
    best_ckpts = [ckpt for ckpt in checkpoints if "best" in ckpt.name.lower()]
    if best_ckpts:
        return best_ckpts[0]
    
    # Try to find last checkpoint
    last_ckpts = [ckpt for ckpt in checkpoints if "last" in ckpt.name.lower()]
    
    # Otherwise find checkpoint with highest epoch number
    epoch_ckpts = []
    for ckpt in checkpoints:
        try:
            if "epoch" in ckpt.name:
                epoch_num = int(ckpt.stem.split("epoch_")[-1].split("_")[0])
                epoch_ckpts.append((epoch_num, ckpt))
        except:
            continue
    
    if epoch_ckpts:
        epoch_ckpts.sort(reverse=True)
        return epoch_ckpts[0][1]
    
    # Fallback to last checkpoint
    if last_ckpts:
        return last_ckpts[0]
    
    return None


def discover_model_checkpoints(
    logs_dir: Path,
    model_names: List[str],
    dataset: str = "autism",
) -> Dict[str, Dict[str, any]]:
    """Discover trained model checkpoints in logs directory.
    
    Args:
        logs_dir: Path to logs directory.
        model_names: List of model names to search for.
        dataset: Dataset name (e.g., "autism", "mental").
        
    Returns:
        Dictionary mapping model names to checkpoint info.
    """
    discovered = {}
    
    train_logs_dir = logs_dir / "train"
    
    if not train_logs_dir.exists():
        print(f"Warning: Training logs directory not found: {train_logs_dir}")
        return discovered
    
    # Search both runs and multiruns directories
    search_dirs = []
    runs_dir = train_logs_dir / "runs"
    multiruns_dir = train_logs_dir / "multiruns"
    
    if runs_dir.exists():
        search_dirs.extend(list(runs_dir.iterdir()))
    
    if multiruns_dir.exists():
        # For multiruns, we need to go one level deeper to find actual run directories
        for multirun_dir in multiruns_dir.iterdir():
            if multirun_dir.is_dir():
                # Each numbered subdirectory (0, 1, 2, ...) is an actual run
                search_dirs.extend([d for d in multirun_dir.iterdir() if d.is_dir()])
    
    # Sort by modification time (newest first) to prefer recent runs
    run_dirs = sorted(search_dirs, key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    
    for model_name in model_names:
        print(f"\nSearching for {model_name} checkpoints...")
        
        # Look for runs matching this model and dataset
        # New naming format: {dataset}_{model}_{timestamp}
        pattern = f"{dataset}_{model_name}"
        
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            
            # First, check if directory name matches the pattern (faster)
            dir_name = run_dir.name
            if dir_name.startswith(pattern):
                print(f"  Found matching directory: {dir_name}")
                best_ckpt = find_best_checkpoint(run_dir)
                
                if best_ckpt:
                    val_acc = extract_val_accuracy_from_checkpoint(best_ckpt)
                    
                    print(f"  ✓ Found checkpoint: {best_ckpt}")
                    if val_acc:
                        print(f"    Validation accuracy: {val_acc:.4f}")
                    
                    discovered[model_name] = {
                        "checkpoint_path": str(best_ckpt),
                        "val_accuracy": val_acc if val_acc else 0.85,  # Default
                        "run_dir": str(run_dir),
                    }
                    break  # Found checkpoint for this model
                continue
            
            # Fallback: Check .hydra/config.yaml for old-style directories
            hydra_config = run_dir / ".hydra" / "config.yaml"
            
            if hydra_config.exists():
                try:
                    with open(hydra_config) as f:
                        config = yaml.safe_load(f)
                    
                    # Check if this is the right model by checking config names
                    model_config_name = config.get("model", {}).get("name", "")
                    data_config_name = config.get("data", {}).get("name", "")
                    
                    # First try matching by config names
                    if model_config_name == model_name and data_config_name == dataset:
                        best_ckpt = find_best_checkpoint(run_dir)
                        
                        if best_ckpt:
                            val_acc = extract_val_accuracy_from_checkpoint(best_ckpt)
                            
                            print(f"  ✓ Found checkpoint: {best_ckpt}")
                            if val_acc:
                                print(f"    Validation accuracy: {val_acc:.4f}")
                            
                            discovered[model_name] = {
                                "checkpoint_path": str(best_ckpt),
                                "val_accuracy": val_acc if val_acc else 0.85,  # Default
                                "run_dir": str(run_dir),
                            }
                            break
                    
                    # Fallback: Check model target (for old configs without name field)
                    model_target = config.get("model", {}).get("_target_", "")
                    
                    # Extract model type from target
                    if model_name in model_target.lower() or model_name in str(config.get("model", {})).lower():
                        # Check dataset
                        data_target = config.get("data", {}).get("_target_", "")
                        if dataset in data_target.lower() or dataset in str(config.get("data", {})).lower():
                            # Found a match!
                            best_ckpt = find_best_checkpoint(run_dir)
                            
                            if best_ckpt:
                                val_acc = extract_val_accuracy_from_checkpoint(best_ckpt)
                                
                                print(f"  ✓ Found checkpoint: {best_ckpt}")
                                if val_acc:
                                    print(f"    Validation accuracy: {val_acc:.4f}")
                                
                                discovered[model_name] = {
                                    "checkpoint_path": str(best_ckpt),
                                    "val_accuracy": val_acc if val_acc else 0.85,  # Default
                                    "run_dir": str(run_dir),
                                }
                                break  # Found checkpoint for this model
                except Exception as e:
                    print(f"    Warning: Error reading config from {hydra_config}: {e}")
                    continue
        
        if model_name not in discovered:
            print(f"  ✗ No checkpoint found for {model_name}")
    
    return discovered


def generate_ensemble_config(
    discovered_checkpoints: Dict[str, Dict[str, any]],
    output_path: Path,
    dataset: str = "autism",
):
    """Generate ensemble configuration YAML file.
    
    Args:
        discovered_checkpoints: Dictionary of discovered checkpoints.
        output_path: Path to save the configuration file.
        dataset: Dataset name.
    """
    # Auto-detect device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "# @package _global_": None,
        "defaults": [
            "_self_",
            {"override /data": dataset},
        ],
        "task_name": "ensemble_eval",
        "tags": ["ensemble", dataset, "voting"],
        "ensemble": {
            "weighting_strategy": "accuracy",
            "device": device,
            "save_predictions": True,
            "generate_plots": True,
            "allow_inconsistent_preprocessing": False,
            "models": []
        }
    }
    
    # Add discovered models
    for model_name, info in discovered_checkpoints.items():
        model_entry = {
            "name": model_name,
            "checkpoint_path": info["checkpoint_path"],
            "val_accuracy": float(info["val_accuracy"]),
        }
        config["ensemble"]["models"].append(model_entry)
    
    # Save configuration
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Generated ensemble configuration: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover trained model checkpoints and generate ensemble configuration."
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Path to logs directory (default: logs)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="autism",
        choices=["autism", "mental"],
        help="Dataset name (default: autism)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["bilstm", "stacked_lstm", "gru", "autoencoder", "vae"],
        help="List of model names to include (default: bilstm stacked_lstm gru autoencoder vae)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ensemble config (default: configs/ensemble/{dataset}_voting_auto.yaml)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    logs_dir = Path(args.logs_dir)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"configs/ensemble/{args.dataset}_voting_auto.yaml")
    
    print("=" * 70)
    print("ENSEMBLE CHECKPOINT DISCOVERY")
    print("=" * 70)
    print(f"Logs directory: {logs_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 70)
    
    # Discover checkpoints
    discovered = discover_model_checkpoints(
        logs_dir=logs_dir,
        model_names=args.models,
        dataset=args.dataset,
    )
    
    if not discovered:
        print("\n✗ No checkpoints found. Please train models first.")
        sys.exit(1)
    
    print(f"\n✓ Found {len(discovered)}/{len(args.models)} model checkpoints.")
    
    # Generate configuration
    generate_ensemble_config(
        discovered_checkpoints=discovered,
        output_path=output_path,
        dataset=args.dataset,
    )
    
    print("\nTo run ensemble evaluation:")
    print(f"  python src/ensemble_eval.py ensemble={output_path.stem}")


if __name__ == "__main__":
    main()
