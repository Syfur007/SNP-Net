# Interpretability Analysis Guide

Complete guide for generating SHAP, Integrated Gradients (IG), and LIME attributions across multiple SNP classification models and datasets for thesis/journal publication.

## Overview

This guide covers:
1. **Generalized explainability methods** working with all 11 model architectures
2. **LIME implementation** for local model-agnostic explanations
3. **Automated analysis pipeline** via Hydra configuration
4. **Cross-architecture robustness analysis** identifying consensus SNPs
5. **Publication-quality visualizations** for thesis/journal submission

## Prerequisites

- Trained model checkpoints in `logs/train/runs/`
- Data files in `data/` directory
- Python libraries: `torch`, `lightning`, `hydra`, `shap`, `captum`, `lime`, `pandas`, `matplotlib`, `seaborn`

Install missing dependencies:
```bash
pip install lime matplotlib-venn
```

## Quick Start

### Option 1: Run Individual Method (Fastest)

For a single checkpoint and dataset with one method:

```bash
# SHAP analysis
python src/shap_explainability.py \
  --checkpoint_path logs/train/runs/experiment_name_date/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv

# Integrated Gradients analysis
python src/integrated_gradients_explainability.py \
  --checkpoint_path logs/train/runs/experiment_name_date/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv

# LIME analysis
python src/lime_explainability.py \
  --checkpoint_path logs/train/runs/experiment_name_date/checkpoints/best.ckpt \
  --data_file data/FinalizedAutismData.csv
```

Output files created:
- `top_{method}_snps.csv` — Ranked SNPs by importance
- `{method}_bar_top20.png` — Bar chart of top-20 SNPs
- `{method}_heatmap_top100.png` — Per-sample heatmap (SHAP/IG only)

### Option 2: Batch Analysis Pipeline (Comprehensive)

Run all methods on all checkpoints and datasets via Hydra:

```bash
python src/interpretability_pipeline.py
```

Default behavior:
- Discovers all best checkpoints from `logs/train/runs/`
- Analyzes across all 6 datasets (Autism, Mental, GSE139294, GSE31276, GSE33355, GSE90073)
- Runs all enabled methods (SHAP, IG, LIME)
- Saves to `outputs/interpretability_analysis/`

#### Customize Analysis

Edit `configs/analysis/interpretability.yaml` or override on CLI:

```bash
# Run only SHAP and IG (skip LIME)
python src/interpretability_pipeline.py \
  methods.lime.enabled=false

# Analyze only Autism dataset
python src/interpretability_pipeline.py \
  datasets='[{name: autism, data_file: data/FinalizedAutismData.csv, config_name: autism}]'

# Use specific checkpoint
python src/interpretability_pipeline.py \
  'checkpoint_selection.checkpoints=[logs/train/runs/my_experiment_2026-02-19/checkpoints/best.ckpt]' \
  checkpoint_selection.mode=custom

# Change output directory
python src/interpretability_pipeline.py \
  analysis.output_base_dir=my_results/interpretability
```

## Method Details

### SHAP (SHapley Additive exPlanations)

**What it does**: Computes feature attributions using gradient-based Shapley values.

**Pros**:
- Theoretically grounded (Shapley values)
- Works with any architecture
- Per-sample explanations aggregated to global importance

**Cons**:
- Slower than IG (~5-10 min per 100 samples with 50 background samples)
- Requires background sampling

**Key parameters**:
- `num_background`: Number of background samples (default: 50; increase for stability)
- `num_test`: Number of test samples to explain (default: 100)

**Example**:
```python
python src/shap_explainability.py \
  --checkpoint_path <path> \
  --data_file data/FinalizedAutismData.csv \
  --num_background 100 \
  --num_test 200
```

### Integrated Gradients (IG)

**What it does**: Computes attributions by integrating gradients along a path from a baseline to the input.

**Pros**:
- Faster than SHAP (~2-5 min per 100 samples)
- Deterministic (no randomness)
- Theoretical guarantees (completeness axiom)

**Cons**:
- Sensitive to baseline choice
- Requires computing gradients through the network

**Key parameters**:
- `baseline_type`: 'zero', 'mean', or 'random' (default: 'mean')
- `n_steps`: Integration steps (default: 50; higher = smoother but slower)
- `num_test`: Number of test samples (default: 100)

**Example**:
```python
python src/integrated_gradients_explainability.py \
  --checkpoint_path <path> \
  --data_file data/FinalizedAutismData.csv \
  --baseline_type mean \
  --n_steps 50
```

### LIME (Local Interpretable Model-agnostic Explanations)

**What it does**: Explains individual predictions by fitting local surrogate models.

**Pros**:
- Model-agnostic (no architectural knowledge needed)
- Provides local explanations (why was THIS sample classified this way?)
- Fast (~30 seconds per sample)

**Cons**:
- Local explanations (not global feature importance)
- Requires aggregation for global importance comparison
- Perturbation-based (slower with larger datasets)

**Key parameters**:
- `num_test`: Number of test samples (default: 100)
- `num_samples_per_test`: Perturbations per sample (default: 100; higher = more stable)

**Example**:
```python
python src/lime_explainability.py \
  --checkpoint_path <path> \
  --data_file data/FinalizedAutismData.csv \
  --num_test 100 \
  --num_samples_per_test 100
```

## Architecture Support

All three methods now work with all 11 model architectures:

- **BiLSTM** ✓
- **Transformer-CNN** ✓
- **Dense** ✓
- **GRU** ✓
- **LSTM** ✓
- **Stacked LSTM** ✓
- **Autoencoder** ✓
- **VAE** ✓
- **DeepPlantCRE** ✓
- **DPCFormer** ✓
- **WheatGP** ✓

No architecture-specific code needed—methods automatically adapt.

## Output Structure

```
outputs/interpretability_analysis/
├── {checkpoint_name}/
│   ├── {dataset_name}/
│   │   ├── shap/
│   │   │   ├── top_shap_snps.csv
│   │   │   ├── shap_bar_top20.png
│   │   │   └── shap_heatmap_top100.png
│   │   ├── ig/
│   │   │   ├── top_ig_snps.csv
│   │   │   ├── ig_bar_top20.png
│   │   │   └── ig_heatmap_top100.png
│   │   └── lime/
│   │       ├── top_lime_snps.csv
│   │       └── lime_bar_top20.png
├── publication_figures/
│   ├── consensus_snps_autism_top30.png
│   ├── consensus_snps_gse139294_top30.png
│   └── ...
├── supplementary_data/
│   ├── consensus_snps_autism_full.csv
│   ├── rank_correlations_autism.csv
│   └── top_k_overlaps_autism.json
└── analysis_config.yaml
```

## Robustness and Consensus Analysis

Identify SNPs that are robustly important across multiple architectures and methods.

### Key Metrics

**Consensus Ratio**: Fraction of architecture/method combinations where an SNP appears in top-K
- High consensus (≥80%) → highly robust for publication
- Medium consensus (50-80%) → reasonably robust
- Low consensus (<50%) → method/architecture-specific

**Rank Correlation**: How well do two methods rank the same SNPs?
- Spearman ρ > 0.7 → strong agreement
- Spearman ρ > 0.5 → moderate agreement
- Spearman ρ < 0.5 → weak agreement

**Top-K Overlap (Jaccard Index)**: Proportion of shared SNPs in top-K lists
- Jaccard > 0.5 → strong agreement
- Jaccard > 0.3 → moderate agreement
- Jaccard < 0.3 → weak agreement

### Computing Robustness Analysis Manually

```python
from src.utils.robustness_analysis import RobustnessAnalyzer

analyzer = RobustnessAnalyzer('outputs/interpretability_analysis')

# Find consensus SNPs
consensus_df = analyzer.compute_consensus_snps(
    checkpoint_dirs=['checkpoint_1', 'checkpoint_2', ...],
    dataset_name='autism',
    top_k=50,
    min_agreement_ratio=0.5
)

# Compare rank correlations
corr_df = analyzer.compute_rank_correlations(
    checkpoint_dirs=['checkpoint_1', 'checkpoint_2', ...],
    dataset_name='autism'
)

# Check top-K overlaps
overlaps = analyzer.compute_top_k_overlap(
    checkpoint_dirs=['checkpoint_1', 'checkpoint_2', ...],
    dataset_name='autism',
    top_k_values=[10, 20, 50]
)
```

## Interactive Exploration

Use the Jupyter notebook for exploration:

```bash
jupyter notebook notebooks/interpretability_explorer.ipynb
```

Features:
- Browse SNP rankings per architecture/method
- Compare top-20 SNPs across SHAP/IG/LIME
- View consensus SNPs with agreement ratios
- Generate publication figures
- Export supplementary data

## Publication Recommendations

### For Thesis

1. **Main figures** (Chapters):
   - Figure A: Top-30 consensus SNPs (bar chart with agreement ratios)
   - Figure B: Method comparison for top architecture (SHAP vs IG vs LIME top-10 overlap)
   - Figure C: Per-sample heatmap (top-100 SNPs across cases/controls) for best model

2. **Supplementary materials** (Appendix):
   - Supplementary Table 1: Full consensus SNPs ranked by agreement (200 SNPs)
   - Supplementary Table 2: Rank correlations between methods
   - Supplementary Table 3: Top-K overlaps (Jaccard indices)
   - Supplementary Figures: Top SNPs per architecture (1 figure per model type)

### For Journal Article

1. **Main figure**:
   - Single integrated visualization (top-20 consensus SNPs with confidence bands)

2. **Supplementary materials**:
   - Consensus rankings (CSV)
   - Method agreement matrix (heatmap)
   - Reproducibility code (this pipeline)

## Troubleshooting

### "No checkpoints found"

**Cause**: The pipeline couldn't locate trained models.

**Solution**:
```bash
# Check if logs directory exists and has checkpoints
ls logs/train/runs/
ls logs/train/runs/*/checkpoints/best.ckpt

# If empty, train a model first
python src/train.py
```

### SHAP/"Out of Memory" error

**Cause**: Too many background samples or test samples.

**Solution**:
```bash
# Reduce background samples
python src/shap_explainability.py --num_background 20 --num_test 50
```

### IG/"NaN attributions" issue

**Cause**: Baseline type producing degenerate gradients.

**Solution**:
```bash
# Try different baseline
python src/integrated_gradients_explainability.py --baseline_type zero
# Or increase integration steps
python src/integrated_gradients_explainability.py --baseline_type mean --n_steps 100
```

### LIME/"No features to explain" error

**Cause**: Feature count mismatch after feature selection.

**Solution**: Ensure feature selection indices are properly saved in checkpoint.

## Performance and Runtime Estimates

### Single Checkpoint, Single Dataset

| Method | Time (100 test samples) | Memory |
|--------|------------------------|--------|
| SHAP   | 5-10 min               | 4-8 GB |
| IG     | 2-5 min                | 2-4 GB |
| LIME   | 1-3 min                | 1-2 GB |

### Batch Analysis

46 checkpoints × 6 datasets:
- SHAP: ~23-46 hours (on single GPU, can parallelize by dataset)
- IG: ~9-23 hours
- LIME: ~4-9 hours

**Recommendation for quick iteration**:
```bash
# Test on 1 checkpoint, 1 dataset first
python src/shap_explainability.py --num_test 20 --num_background 10
# Then run full pipeline
python src/interpretability_pipeline.py methods.shap.num_test=100
```

## Next Steps

1. **Run quick test** (1 checkpoint, 1 dataset):
   ```bash
   python src/shap_explainability.py \
     --checkpoint_path logs/train/runs/latest_run/checkpoints/best.ckpt \
     --data_file data/FinalizedAutismData.csv \
     --output_dir outputs/test_shap
   ```

2. **Explore results**:
   ```bash
   jupyter notebook notebooks/interpretability_explorer.ipynb
   ```

3. **Run batch analysis** (all checkpoints, all datasets):
   ```bash
   python src/interpretability_pipeline.py
   ```

4. **Generate publication figures**:
   - Run cells in `notebooks/interpretability_explorer.ipynb` (Section 6-7)
   - Or use `src/utils/robustness_analysis.py` directly

## References

- **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- **Integrated Gradients**: Sundararajan et al. (2017) - "Axiomatic Attribution for Deep Networks"
- **LIME**: Ribeiro et al. (2016) - "Why Should I Trust You?: Explaining the Predictions of Any Classifier"
