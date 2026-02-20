#!/bin/bash

# Interpretability Analysis - Quick Reference Commands

# ==============================================================================
# PART A: Quick Tests (Run these first to verify setup)
# ==============================================================================

# Test SHAP on single checkpoint
echo "Testing SHAP analysis..."
python src/shap_explainability.py \
  --checkpoint_path logs/train/runs/$(ls -t logs/train/runs | head -1)/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv \
  --num_test 20 --num_background 10 \
  --output_dir outputs/quick_test/shap

# Test IG on same checkpoint
echo "Testing Integrated Gradients..."
python src/integrated_gradients_explainability.py \
  --checkpoint_path logs/train/runs/$(ls -t logs/train/runs | head -1)/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv \
  --num_test 20 \
  --output_dir outputs/quick_test/ig

# Test LIME on same checkpoint
echo "Testing LIME..."
python src/lime_explainability.py \
  --checkpoint_path logs/train/runs/$(ls -t logs/train/runs | head -1)/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv \
  --num_test 20 \
  --output_dir outputs/quick_test/lime

# ==============================================================================
# PART B: Full Analysis - Single Method
# ==============================================================================

# SHAP analysis on latest checkpoint (100 samples, 50 background)
python src/shap_explainability.py \
  --checkpoint_path logs/train/runs/$(ls -t logs/train/runs | head -1)/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv \
  --num_test 100 --num_background 50

# IG analysis on latest checkpoint (100 samples, 50 steps, mean baseline)
python src/integrated_gradients_explainability.py \
  --checkpoint_path logs/train/runs/$(ls -t logs/train/runs | head -1)/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv \
  --num_test 100 --baseline_type mean --n_steps 50

# LIME analysis on latest checkpoint (100 samples, 100 perturbations)
python src/lime_explainability.py \
  --checkpoint_path logs/train/runs/$(ls -t logs/train/runs | head -1)/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv \
  --num_test 100 --num_samples_per_test 100

# ==============================================================================
# PART C: Batch Analysis Pipeline
# ==============================================================================

# Run batch pipeline with all defaults (all checkpoints, all datasets)
# ⏱️ Estimated time: SHAP ~23 hours, IG ~9 hours, LIME ~4 hours (staggered)
python src/interpretability_pipeline.py

# Run only SHAP (skip IG and LIME for speed)
python src/interpretability_pipeline.py \
  methods.ig.enabled=false \
  methods.lime.enabled=false

# Run only on Autism dataset
python src/interpretability_pipeline.py \
  'datasets=[{name: autism, data_file: data/FinalizedAutismData.csv, config_name: autism}]'

# Use specific checkpoints instead of auto-discovery
python src/interpretability_pipeline.py \
  'checkpoint_selection.checkpoints=[logs/train/runs/experiment_name_2026-02-19/checkpoints/best.ckpt]' \
  checkpoint_selection.mode=custom

# Change output directory
python src/interpretability_pipeline.py \
  analysis.output_base_dir=results/my_analysis

# ==============================================================================
# PART D: Interactive Exploration
# ==============================================================================

# Launch Jupyter notebook for exploration and figure generation
jupyter notebook notebooks/interpretability_explorer.ipynb

# ==============================================================================
# PART E: Robustness Analysis (Manual)
# ==============================================================================

# Python snippet to compute consensus SNPs and rank correlations
python << 'EOF'
from src.utils.robustness_analysis import RobustnessAnalyzer
import glob

# Initialize
analyzer = RobustnessAnalyzer('outputs/interpretability_analysis')

# Get all checkpoint directories
checkpoint_dirs = [d.name for d in Path('outputs/interpretability_analysis').iterdir() 
                   if d.is_dir() and d.name not in ['figures', 'data']]

# Compute consensus for autism dataset
consensus_df = analyzer.compute_consensus_snps(
    checkpoint_dirs=checkpoint_dirs,
    dataset_name='autism',
    methods=['shap', 'ig', 'lime'],
    top_k=50,
    min_agreement_ratio=0.3
)

print(f"Found {len(consensus_df)} consensus SNPs")
print("\nTop 20:")
print(consensus_df.head(20).to_string(index=False))

# Save
consensus_df.to_csv('outputs/consensus_snps_autism.csv', index=False)
print("\n✓ Saved to outputs/consensus_snps_autism.csv")
EOF

# ==============================================================================
# PART F: Useful Utilities
# ==============================================================================

# Find latest checkpoint
LATEST_CKPT=$(ls -t logs/train/runs/*/checkpoints/best.ckpt | head -1)
echo "Latest checkpoint: $LATEST_CKPT"

# Find all checkpoints
echo "All available checkpoints:"
find logs/train/runs -name best.ckpt | sort

# Count results generated
echo "SHAP results:"
find outputs/interpretability_analysis -name "top_shap_snps.csv" | wc -l

echo "IG results:"
find outputs/interpretability_analysis -name "top_ig_snps.csv" | wc -l

echo "LIME results:"
find outputs/interpretability_analysis -name "top_lime_snps.csv" | wc -l

# ==============================================================================
# PART G: Dataset Mappings
# ==============================================================================

# Dataset files for reference:
# Autism:   data/FinalizedAutismData.csv
# Mental:   data/FinalizedMentalData.csv
# GSE139294: data/Finalized_GSE139294.csv
# GSE31276: data/Finalized_GSE31276.csv
# GSE33355: data/Finalized_GSE33355.csv
# GSE90073: data/Finalized_GSE90073.csv

# ==============================================================================
# PART H: Common Options
# ==============================================================================

# SHAP options:
#   --num_background {int}   Number of background samples (default: 50)
#   --num_test {int}         Number of test samples (default: 100)
#   --top_k_bar {int}        Top-K for bar plot (default: 20)
#   --top_k_heatmap {int}    Top-K for heatmap (default: 100)

# IG options:
#   --num_test {int}         Number of test samples (default: 100)
#   --baseline_type {str}    'zero', 'mean', or 'random' (default: 'mean')
#   --n_steps {int}          Integration steps (default: 50)
#   --top_k_bar {int}        Top-K for bar plot (default: 20)

# LIME options:
#   --num_test {int}         Number of test samples (default: 100)
#   --num_samples_per_test   Perturbations per sample (default: 100)
#   --top_k_bar {int}        Top-K for bar plot (default: 20)

echo "✓ Quick reference loaded. See comments above for command examples."
