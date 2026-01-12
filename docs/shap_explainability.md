# SHAP Explainability for SNP Classification Models

This document describes the implementation and usage of SHAP (SHapley Additive exPlanations) analysis for explaining predictions from deep learning models trained on SNP data.

## Overview

The SHAP explainability system provides model-agnostic interpretability by computing feature attribution scores for individual predictions. This helps identify which SNPs contribute most to the model's classification decisions.

### Supported Models
- **Bi-LSTM**: Bidirectional LSTM with sequence modeling
- **Stacked LSTM**: Multi-layer hierarchical LSTM
- **GRU**: Gated Recurrent Unit
- **Dense/MLP**: Fully-connected neural networks
- **Transformer-CNN**: Transformer with CNN components
- **Autoencoder + MLP**: Autoencoder with classifier head
- **VAE + MLP**: Variational Autoencoder with classifier head

### SHAP Attribution Formula

For each test sample and feature, SHAP computes:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_{S \cup \{j\}}(x_{S \cup \{j\}}) - f_S(x_S)]$$

where:
- $\phi_j$ = SHAP value for feature $j$ (SNP importance)
- $F$ = set of all features
- $S$ = subset of features
- $f$ = model prediction function
- Aggregated across test samples to rank SNP importance

## Features

✅ **GradientExplainer Integration**: Uses efficient gradient-based SHAP computation for deep learning models

✅ **Automatic Checkpoint Loading**: Loads best checkpoint from training with full model configuration

✅ **SNP Importance Ranking**: Aggregates and ranks all SNPs by mean absolute SHAP values

✅ **Publication-Ready Visualizations**: Bar plots of top SNPs, heatmaps showing sample-level attributions

✅ **Comprehensive Output**: CSV rankings, statistics, and multiple visualization formats

✅ **Configurable Sample Sizes**: Adjustable number of background and test samples for speed/accuracy tradeoff

✅ **Multi-Dataset Support**: Works with autism, mental health, and mayocardial datasets

## Directory Structure

```
src/
├── shap_explainability.py         # Main SHAP analysis script
└── integrated_gradients_explainability.py  # Alternative IG-based explainability

data/
├── FinalizedAutismData.csv        # Autism dataset
├── FinalizedMentalData.csv        # Mental health dataset
└── FinalizedMayocardialData.csv   # Mayocardial dataset

logs/train/runs/{experiment}/
├── checkpoints/
│   └── best.ckpt                  # Best model checkpoint
└── shap_results/                  # SHAP output directory (auto-created)
    ├── top_shap_snps.csv          # Ranked SNP importance list
    ├── shap_bar_top20.png         # Bar plot of top SNPs
    ├── shap_heatmap_top50.png     # Sample-level attribution heatmap
    └── shap_analysis.log          # Detailed execution log
```

## Installation

No additional dependencies required beyond the base SNP-Net environment. SHAP is included in `requirements.txt`.

## Usage

### Step 1: Train a Model

First, train a model using an experiment configuration:

```bash
# Train Bi-LSTM on autism dataset
python src/train.py experiment=autism_bilstm

# Train Dense network on mental health dataset
python src/train.py experiment=mental_dense

# Train any supported architecture
python src/train.py experiment=autism_lstm
```

This will save checkpoints to `logs/train/runs/{timestamp}/checkpoints/best.ckpt`.

### Step 2: Run SHAP Explainability

After training completes, run SHAP analysis on the best checkpoint:

```bash
# Basic usage with default parameters
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{timestamp}/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv

# Advanced usage with custom parameters
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/2025-01-12_10-30-45/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --num_background 100 \
    --num_test 200 \
    --top_k_bar 30 \
    --top_k_heatmap 100 \
    --output_dir custom_output/
```

### Step 3: View Results

Results are saved to the output directory (default: same directory as checkpoint):

```
shap_results/
├── top_shap_snps.csv              # Ranked list of all SNPs with importance scores
├── shap_bar_top20.png             # Bar plot showing top N most important SNPs
└── shap_heatmap_top50.png         # Heatmap of SNP importance across samples
```

## Command-Line Arguments

### Required Arguments

- `--checkpoint_path`: Path to the trained model checkpoint (.ckpt file)
  ```bash
  --checkpoint_path logs/train/runs/{experiment}/checkpoints/best.ckpt
  ```

### Optional Arguments

- `--data_file`: Path to the dataset CSV file (default: `data/FinalizedAutismData.csv`)
  ```bash
  --data_file data/FinalizedMentalData.csv
  ```

- `--output_dir`: Output directory for results (default: same as checkpoint directory)
  ```bash
  --output_dir custom_results/shap_analysis/
  ```

- `--num_background`: Number of background samples for SHAP baseline (default: 50)
  ```bash
  --num_background 100  # More samples = more accurate but slower
  ```

- `--num_test`: Number of test samples to explain (default: 100)
  ```bash
  --num_test 200  # More samples = more robust rankings
  ```

- `--device`: Device for computation (default: auto-detect CUDA/CPU)
  ```bash
  --device cuda      # Force GPU
  --device cpu       # Force CPU
  ```

- `--top_k_bar`: Number of top SNPs to show in bar plot (default: 20)
  ```bash
  --top_k_bar 30
  ```

- `--top_k_heatmap`: Number of top SNPs to show in heatmap (default: 50)
  ```bash
  --top_k_heatmap 100
  ```

## Configuration Examples

### Quick Analysis (Fast)

For rapid iteration during development:

```bash
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{timestamp}/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --num_background 20 \
    --num_test 50 \
    --top_k_bar 10 \
    --top_k_heatmap 20
```

**Expected runtime**: ~2-5 minutes on GPU

### Standard Analysis (Recommended)

For balanced speed and accuracy:

```bash
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{timestamp}/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --num_background 50 \
    --num_test 100 \
    --top_k_bar 20 \
    --top_k_heatmap 50
```

**Expected runtime**: ~5-10 minutes on GPU

### Comprehensive Analysis (Publication)

For high-quality results for publications:

```bash
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{timestamp}/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --num_background 200 \
    --num_test 500 \
    --top_k_bar 50 \
    --top_k_heatmap 100 \
    --device cuda
```

**Expected runtime**: ~20-40 minutes on GPU

## Output Interpretation

### Ranked SNP List (top_shap_snps.csv)

CSV file with columns:
- `SNP_ID`: Identifier for each SNP
- `Mean_Abs_SHAP`: Mean absolute SHAP value across all test samples
- `Std_SHAP`: Standard deviation of SHAP values (variability across samples)
- `Rank`: Importance ranking (1 = most important)

```csv
SNP_ID,Mean_Abs_SHAP,Std_SHAP,Rank
rs7234567,0.0245,0.0123,1
rs8901234,0.0198,0.0089,2
rs4567890,0.0176,0.0102,3
...
```

**Interpretation**:
- **High Mean_Abs_SHAP**: SNP strongly influences model predictions
- **High Std_SHAP**: SNP importance varies across samples (may be context-dependent)
- **Low Std_SHAP**: SNP importance is consistent across samples

### Bar Plot (shap_bar_top20.png)

Horizontal bar chart showing top K most important SNPs.

**Features**:
- SNPs ordered by importance (most important at top)
- Bar length represents mean absolute SHAP value
- Color-coded by importance magnitude
- Includes numeric importance scores

**Usage**: Identify the most globally important SNPs for the classification task.

### Heatmap (shap_heatmap_top50.png)

2D heatmap showing SNP importance across individual test samples.

**Axes**:
- **X-axis**: Test samples (colored by true label)
- **Y-axis**: Top K SNPs (ordered by importance)

**Color Scale**:
- **Red**: Positive SHAP values (pushes prediction toward class 1)
- **Blue**: Negative SHAP values (pushes prediction toward class 0)
- **White**: Near-zero SHAP values (minimal contribution)

**Usage**: 
- Identify sample-specific important SNPs
- Detect patterns in SNP importance across classes
- Discover subgroups with different important features

### Console Output

During execution, you'll see:

```
================================================================================
LOADING MODEL AND DATA
================================================================================
✓ Loading checkpoint from: logs/train/runs/.../checkpoints/best.ckpt
✓ Model loaded: BiLSTM
✓ Validation Accuracy: 0.8542

✓ Loaded data from: data/FinalizedAutismData.csv
✓ Data shape: (1234, 5678)
✓ Test samples: 247

================================================================================
COMPUTING SHAP VALUES
================================================================================
✓ Using device: cuda
✓ Background samples: 50
✓ Test samples to explain: 100
✓ Computing SHAP attributions...
  [Progress: ████████████████████] 100%

✓ SHAP values computed: shape (100, 1000)

================================================================================
GENERATING VISUALIZATIONS
================================================================================
✓ SNP importance aggregated and ranked
✓ Bar plot saved to: shap_bar_top20.png
✓ Heatmap saved to: shap_heatmap_top50.png

================================================================================
SHAP EXPLAINABILITY ANALYSIS COMPLETE
================================================================================
Results saved to: logs/train/runs/.../shap_results/
  - Ranked SNP list: top_shap_snps.csv
  - Bar plot: shap_bar_top20.png
  - Heatmap: shap_heatmap_top50.png
================================================================================

Summary Statistics:
  Total SNPs analyzed: 1000
  Test samples explained: 100
  Mean importance (top 20): 0.018456
  Mean importance (all): 0.003421
  Importance range: [0.000012, 0.024567]
```

## Advanced Usage

### Multiple Datasets

Run SHAP analysis on different datasets:

```bash
# Autism dataset
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{autism_exp}/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv

# Mental health dataset
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{mental_exp}/checkpoints/best.ckpt \
    --data_file data/FinalizedMentalData.csv

# Mayocardial dataset
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/{mayo_exp}/checkpoints/best.ckpt \
    --data_file data/FinalizedMayocardialData.csv
```

### Batch Processing Multiple Models

Create a bash script to analyze multiple models:

```bash
#!/bin/bash
# analyze_all_models.sh

MODELS=(
    "logs/train/runs/bilstm_run1/checkpoints/best.ckpt"
    "logs/train/runs/lstm_run1/checkpoints/best.ckpt"
    "logs/train/runs/gru_run1/checkpoints/best.ckpt"
)

for model in "${MODELS[@]}"; do
    echo "Analyzing $model..."
    python src/shap_explainability.py \
        --checkpoint_path "$model" \
        --data_file data/FinalizedAutismData.csv \
        --num_background 50 \
        --num_test 100
done
```

### Comparative Analysis

Compare SNP rankings across different models:

```python
import pandas as pd

# Load SHAP results from multiple models
bilstm_snps = pd.read_csv('logs/train/.../bilstm/shap_results/top_shap_snps.csv')
lstm_snps = pd.read_csv('logs/train/.../lstm/shap_results/top_shap_snps.csv')
gru_snps = pd.read_csv('logs/train/.../gru/shap_results/top_shap_snps.csv')

# Merge on SNP_ID and compare rankings
merged = bilstm_snps[['SNP_ID', 'Rank']].merge(
    lstm_snps[['SNP_ID', 'Rank']], 
    on='SNP_ID', 
    suffixes=('_BiLSTM', '_LSTM')
).merge(
    gru_snps[['SNP_ID', 'Rank']], 
    on='SNP_ID'
).rename(columns={'Rank': 'Rank_GRU'})

# Find consensus top SNPs (top 10 in all models)
consensus = merged[
    (merged['Rank_BiLSTM'] <= 10) & 
    (merged['Rank_LSTM'] <= 10) & 
    (merged['Rank_GRU'] <= 10)
]
print(f"Consensus important SNPs: {len(consensus)}")
print(consensus)
```

### Custom Output Organization

Organize outputs by experiment:

```bash
# Create organized directory structure
mkdir -p results/shap_analysis/{bilstm,lstm,gru}

# Run with custom output directories
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/bilstm/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --output_dir results/shap_analysis/bilstm/

python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/lstm/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --output_dir results/shap_analysis/lstm/
```

## Troubleshooting

### Error: "Checkpoint file not found"

**Cause**: Invalid or incorrect checkpoint path.

**Solution**: 
```bash
# Find your checkpoint
find logs/train/runs -name "best.ckpt" -type f

# Use the full path
python src/shap_explainability.py \
    --checkpoint_path logs/train/runs/2025-01-12_10-30-45/checkpoints/best.ckpt \
    --data_file data/FinalizedAutismData.csv
```

### Error: "Data file not found"

**Cause**: Dataset path is incorrect.

**Solution**: 
```bash
# Check available datasets
ls data/*.csv

# Use correct path
python src/shap_explainability.py \
    --checkpoint_path logs/train/.../best.ckpt \
    --data_file data/FinalizedMentalData.csv  # Correct filename
```

### Error: "CUDA out of memory"

**Cause**: GPU memory insufficient for SHAP computation.

**Solution 1** - Reduce sample sizes:
```bash
python src/shap_explainability.py \
    --checkpoint_path logs/train/.../best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --num_background 20 \
    --num_test 50
```

**Solution 2** - Use CPU:
```bash
python src/shap_explainability.py \
    --checkpoint_path logs/train/.../best.ckpt \
    --data_file data/FinalizedAutismData.csv \
    --device cpu
```

### Error: "Model architecture mismatch"

**Cause**: Checkpoint contains a model architecture not compatible with SHAP wrapper.

**Solution**: Ensure you're using a supported model architecture (BiLSTM, LSTM, GRU, Dense, etc.). Check the model configuration in your training experiment config.

### Warning: "Using subset of test data"

**Cause**: Requested more test samples than available in test set.

**Solution**: This is informational. SHAP will use all available test samples. To see the exact number:
```bash
# Check test set size in output
python src/shap_explainability.py ... | grep "Test samples:"
```

### Slow Execution

**Cause**: SHAP computation is inherently expensive.

**Solutions**:
1. Reduce background samples (minimal accuracy impact):
   ```bash
   --num_background 20
   ```

2. Reduce test samples (may affect ranking robustness):
   ```bash
   --num_test 50
   ```

3. Use GPU acceleration:
   ```bash
   --device cuda
   ```

4. Expected runtimes:
   - GPU: 5-10 minutes (standard settings)
   - CPU: 20-40 minutes (standard settings)

## Performance Expectations

### Computation Time

| Configuration | GPU (RTX 3090) | CPU (12-core) |
|---------------|----------------|---------------|
| Quick (20/50) | ~2 min | ~8 min |
| Standard (50/100) | ~5 min | ~20 min |
| Comprehensive (200/500) | ~25 min | ~90 min |

### Memory Requirements

- **GPU**: 4-8 GB VRAM (depends on model size)
- **CPU**: 8-16 GB RAM
- **Disk**: <100 MB per analysis (outputs)

### Accuracy vs. Speed Tradeoff

- **num_background**: 
  - ↑ More samples → More accurate baseline, slower computation
  - Recommended: 50-100 for standard analysis
  
- **num_test**: 
  - ↑ More samples → More robust rankings, slower computation
  - Recommended: 100-200 for standard analysis

## Biological Interpretation

### Identifying Candidate SNPs

Top-ranked SNPs from SHAP analysis are candidates for:
1. **Functional validation**: Experimental verification of biological role
2. **Literature review**: Check if SNP is in known disease-associated genes
3. **Pathway analysis**: Map to biological pathways and gene networks
4. **Clinical relevance**: Assess potential for diagnostic/prognostic biomarkers

### Cross-Referencing with Databases

Use top SNPs to query:
- **dbSNP**: rs identifiers and allele frequencies
- **GWAS Catalog**: Known disease associations
- **Ensembl**: Gene annotations and functional consequences
- **GTEx**: Expression quantitative trait loci (eQTLs)
- **ClinVar**: Clinical significance

### Example Workflow

```python
# Load top SNPs
import pandas as pd
top_snps = pd.read_csv('shap_results/top_shap_snps.csv')

# Get top 20 SNPs
candidates = top_snps.head(20)['SNP_ID'].tolist()

# Export for external analysis
with open('candidate_snps_for_validation.txt', 'w') as f:
    f.write('\n'.join(candidates))

print(f"Top 20 candidate SNPs: {candidates}")
```

## Comparison with Other Explainability Methods

### SHAP vs. Integrated Gradients

| Aspect | SHAP | Integrated Gradients |
|--------|------|---------------------|
| **Theoretical basis** | Game theory (Shapley values) | Axiomatic attribution |
| **Computation** | Sampling-based | Path integration |
| **Speed** | Moderate | Fast |
| **Baseline** | Multiple background samples | Single baseline (zeros) |
| **Consistency** | Guaranteed | Guaranteed |
| **Implementation** | `shap_explainability.py` | `integrated_gradients_explainability.py` |

**When to use SHAP**:
- Need robust importance estimates
- Multiple baselines required
- Standard in XAI literature

**When to use Integrated Gradients**:
- Speed is critical
- Single baseline appropriate
- Gradient-based attribution preferred

## API Reference

### BiLSTMShapWrapper

Wrapper class for ensuring SHAP-compatible model forward pass:

```python
from src.shap_explainability import BiLSTMShapWrapper

wrapper = BiLSTMShapWrapper(
    model=trained_model,
    seq_len=10,
    window_size=100
)

# Compatible with SHAP explainers
explainer = shap.GradientExplainer(wrapper, background_data)
```

### load_checkpoint()

Load model and configuration from checkpoint:

```python
from src.shap_explainability import load_checkpoint

model, checkpoint = load_checkpoint(
    checkpoint_path="logs/train/.../best.ckpt"
)

print(f"Model type: {checkpoint['hyper_parameters']['_target_']}")
print(f"Validation accuracy: {checkpoint['val/acc']}")
```

### aggregate_and_rank_snps()

Aggregate SHAP values and rank SNPs:

```python
from src.shap_explainability import aggregate_and_rank_snps

snp_importance_df = aggregate_and_rank_snps(
    shap_values=shap_values,  # Shape: (num_samples, num_features)
    snp_ids=snp_identifiers,   # List of SNP IDs
    output_path="rankings.csv"
)
```

### create_bar_plot()

Generate bar plot of top SNPs:

```python
from src.shap_explainability import create_bar_plot

create_bar_plot(
    snp_importance_df=rankings,
    output_path="bar_plot.png",
    top_k=20
)
```

### create_heatmap()

Generate heatmap of sample-level attributions:

```python
from src.shap_explainability import create_heatmap

create_heatmap(
    shap_values=shap_values,
    test_labels=labels,
    snp_importance_df=rankings,
    output_path="heatmap.png",
    top_k=50,
    num_samples=30
)
```

## Integration with Ensemble Models

To explain ensemble predictions, analyze each constituent model separately:

```bash
# Run SHAP on each ensemble member
for model in bilstm stacked_lstm gru autoencoder vae; do
    python src/shap_explainability.py \
        --checkpoint_path logs/train/runs/${model}/checkpoints/best.ckpt \
        --data_file data/FinalizedAutismData.csv \
        --output_dir results/shap_ensemble/${model}/
done

# Aggregate results (custom script)
python scripts/aggregate_ensemble_shap.py \
    --shap_dirs results/shap_ensemble/*/ \
    --output results/shap_ensemble/consensus_snps.csv
```

## Citation

If you use SHAP explainability in your research, please cite:

```bibtex
@software{snp_net_shap,
  title={SHAP Explainability for SNP Classification Models},
  author={Your Name},
  year={2026},
  url={https://github.com/Syfur007/SNP-Net}
}

@inproceedings{lundberg2017unified,
  title={A unified approach to interpreting model predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4765--4774},
  year={2017}
}
```

## Related Tools

### Integrated Gradients Explainability

Alternative attribution method:
```bash
python src/integrated_gradients_explainability.py \
    --checkpoint_path logs/train/.../best.ckpt \
    --data_file data/FinalizedAutismData.csv
```

See [integrated_gradients_explainability.py](../src/integrated_gradients_explainability.py) for details.

### Feature Selection

Pre-training feature selection that complements explainability:
```bash
python src/train.py experiment=autism_lstm \
    data.feature_selection.method=amgm_cosine \
    data.feature_selection.k_features=1000
```

See [feature_selection.md](feature_selection.md) for details.

## Contributing

To extend the SHAP explainability system:

1. **Add new explainer types**: Implement KernelExplainer, TreeExplainer wrappers
2. **Add new visualizations**: Extend plotting functions (waterfall plots, force plots)
3. **Add statistical tests**: Implement significance testing for SNP importance
4. **Add biological enrichment**: Integrate pathway/gene ontology analysis

## License

Same license as SNP-Net project.
