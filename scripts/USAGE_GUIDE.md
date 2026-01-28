# Generic Dataset Split Script - Usage Guide

## Overview
The `generic_split_dataset.py` is a reusable, production-ready template for performing stratified 70/20/10 train/validation/test splits on any genomic or machine learning dataset.

## Features
- **Stratified Splitting**: Maintains class distribution across all three splits
- **Memory Efficient**: Uses garbage collection for large files
- **Reproducible**: Fixed random seed ensures consistent results
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Error Handling**: Comprehensive input validation and error reporting
- **Flexible**: Works with any dataset in the required format

## Input Format Requirements
- **CSV file** with:
  - Rows: SNP/feature IDs (index column)
  - Columns: Sample IDs (plus one row for labels)
  - **Last row**: Contains class labels (e.g., "control", "case")
  - Values: Numeric features (genotypes, gene expressions, etc.)

Example structure:
```
ID_REF,Sample1,Sample2,Sample3,...,SampleN
SNP_001,0,1,2,...,1
SNP_002,1,1,0,...,2
...
SNP_N,2,0,1,...,1
classification,control,case,control,...,case
```

## Basic Usage

### Command Line
```bash
python generic_split_dataset.py <input_csv> <output_dir> <label_row> [--seed 12345]
```

### Arguments
- `input_csv`: Path to input CSV file
- `output_dir`: Directory to save split files (created if doesn't exist)
- `label_row`: Name of the row containing class labels (e.g., "classification")
- `--seed`: Optional random seed (default: 12345)

### Examples

**Example 1: GSE90073 dataset**
```bash
python generic_split_dataset.py data/Finalized_GSE90073.csv data/splits_gse90073 classification
```

**Example 2: GSE139294 dataset**
```bash
python generic_split_dataset.py data/Finalized_GSE139294.csv data/splits_gse139294 classification
```

**Example 3: Custom seed for different split**
```bash
python generic_split_dataset.py data/MyDataset.csv data/my_splits label_column --seed 42
```

## Output

The script generates **three CSV files**:

1. **`{dataset_name}_train.csv`**: 70% of samples
2. **`{dataset_name}_val.csv`**: 20% of samples
3. **`{dataset_name}_test.csv`**: 10% of samples

Each file contains:
- All features as columns (same as input)
- One additional "label" column at the end
- Class labels preserved from original data

### Output Format
```
Feature1,Feature2,Feature3,...,FeatureN,label
0,1,2,...,1,case
1,1,0,...,2,control
...
```

## Split Verification

The script automatically prints a summary:
```
============================================================
STRATIFIED SPLIT SUMMARY
============================================================
Total samples: 106
Training:   74 samples (69.8%)
Validation: 21 samples (19.8%)
Test:       11 samples (10.4%)

Class distribution (stratified across splits):
Training:    {0: 27, 1: 47}
Validation:  {0: 8, 1: 13}
Test:        {0: 4, 1: 7}
============================================================
```

## Key Advantages

### 1. Stratified Splitting
- Maintains exact class proportions in each split
- Critical for imbalanced datasets
- Example: If dataset has 63% case / 37% control, each split will too

### 2. Exact Proportions
- Train: 70% (using `int(total * 0.70)` for exactness)
- Validation: 20% of remaining
- Test: 10% of remaining
- Example: 106 samples → 74/21/11 (69.8%/19.8%/10.4%)

### 3. Production Ready
- Handles missing data gracefully
- Validates inputs before processing
- Memory-efficient for large datasets
- Cross-platform compatibility

## Advanced Usage

### Python Script Integration
```python
from generic_split_dataset import load_and_prepare_data, perform_stratified_split

# Load data
X, y, y_names = load_and_prepare_data('data/mydata.csv', 'classification')

# Create splits
X_train, X_val, X_test, y_train, y_val, y_test = perform_stratified_split(
    X, y, random_seed=42
)
```

## Troubleshooting

### Error: "Label row 'X' not found in CSV"
- Check the exact label row name in your CSV
- Row names are case-sensitive
- Verify the label row is actually in the file

### Error: "Memory error" with large files
- The script uses garbage collection but still needs RAM
- Try running on a machine with more memory
- Consider processing file outside VS Code terminal

### Error: "Mixed types" warning
- Normal for large genomic datasets
- Use `low_memory=False` in pandas read (already done)
- Does not affect results

## Performance Benchmarks

### GSE90073 (106 samples, 541K features)
- Load time: ~2-3 seconds
- Split time: <1 second
- Total output: ~344 MB (3 files)
- Memory usage: ~2 GB peak

### GSE139294 (166 samples, 659K features)
- Load time: ~30 seconds
- Split time: ~2 seconds
- Total output: ~530 MB (3 files)
- Memory usage: ~4 GB peak

## Quality Assurance

The script includes:
- Input validation (file exists, label row exists)
- Output verification (all files created successfully)
- Class distribution checking (stratification verification)
- Progress reporting (detailed logging)

## Contact & Support

For issues or modifications, check:
- `GENERIC_SPLIT_README.md` - Comprehensive technical documentation
- `QUICK_REFERENCE.md` - Quick command examples
- `DATASET_SPLIT_PACKAGE.md` - Package overview and best practices
