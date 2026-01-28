#!/usr/bin/env python
"""Simple test script for Mayocardial Autoencoder model with 5-fold CV (no Hydra CLI args)."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Test the mayocardial autoencoder model with 5-fold cross validation."""
    
    # Get the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Build the config programmatically (no Hydra CLI parsing)
    config = {
        # Experiment config
        'experiment': 'mayocardial_autoencoder',
        'seed': 12345,
        'tags': ['mayocardial', 'autoencoder', 'case_control'],
        'optimized_metric': 'val/acc',
        
        # Data config - 5-fold cross validation
        'data': {
            '_target_': 'src.data.datamodule.DataModule',
            'data_file': 'data/Finalized_GSE90073.csv',
            'batch_size': 32,
            'num_workers': 0,
            'normalize': True,
            'has_header': True,
            'has_index': True,
            'num_folds': 5,  # 5-fold cross validation
            'current_fold': 0,
        },
        
        # Model config - autoencoder
        'model': {
            'net': {
                '_target_': 'src.models.components.autoencoder.Autoencoder',
                'input_dim': 1000,
                'latent_dim': 32,
                'hidden_dims': [128, 64],
            },
        },
        
        # Trainer config
        'trainer': {
            'min_epochs': 2,
            'max_epochs': 5,  # Quick test with 5 epochs
            'accelerator': 'cpu',
            'gradient_clip_val': 0.5,
        },
        
        # Callbacks
        'callbacks': {
            'early_stopping': {
                'monitor': 'val/loss',
                'patience': 3,
                'mode': 'min',
            },
        },
        
        # Logger
        'logger': {
            'wandb': {
                'name': 'Mayocardial-Autoencoder-Test-5Fold',
            },
        },
        
        # Paths
        'paths': {
            'root_dir': str(project_root),
            'data_dir': str(project_root / 'data'),
            'log_dir': str(project_root / 'logs'),
            'output_dir': str(project_root / 'logs' / 'train'),
        },
    }
    
    print("=" * 80)
    print("Testing Mayocardial Autoencoder with 5-Fold Cross Validation")
    print("=" * 80)
    print(f"\nConfig Summary:")
    print(f"  - Data: Finalized_GSE90073.csv")
    print(f"  - Cross Validation: 5-fold")
    print(f"  - Model: Autoencoder (latent_dim=32)")
    print(f"  - Epochs: 2-5 (with early stopping)")
    print(f"  - Batch size: 32")
    print()
    
    try:
        # Import here after sys.path is set
        from src.train import train
        from omegaconf import OmegaConf
        
        # Convert dict to OmegaConf for compatibility
        cfg = OmegaConf.create(config)
        
        print("Starting training...\n")
        metric_dict, object_dict = train(cfg=cfg)
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"\nFinal Metrics:")
        for key, value in metric_dict.items():
            print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
