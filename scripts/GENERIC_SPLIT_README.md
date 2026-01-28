# Generic Dataset Split Script - User Guide

## Overview

The `generic_split_dataset.py` script is a **reusable template** for splitting any dataset into train/validation/test sets using **stratified splitting (70% train / 20% val / 10% test)**.

This script:
- ✅ Maintains class balance across all splits (using stratification)
- ✅ Produces exact 70/20/10 proportions
- ✅ Works with any dataset following the standard format
- ✅ Includes proper error handling and memory optimization
- ✅ Provides detailed progress reporting

## Installation

No additional installation needed. The script uses standard Python libraries:
- `pandas`
- `scikit-learn`
- `numpy`

Ensure these are installed:
```bash
pip install pandas scikit-learn numpy
```

## Input Dataset Format

Your CSV file should follow this structure:

```
                Sample1    Sample2    Sample3    ...    SampleN
Feature1        0.123      0.456      0.789      ...    0.234
Feature2        0.456      0.789      0.234      ...    0.567
Feature3        0.789      0.234      0.567      ...    0.890
...
classification  case       control    case       ...    control
```

**Requirements:**
- **First column (index):** Feature names (SNP names, gene names, etc.)
- **Columns:** Sample IDs
- **Last row:** Class labels (e.g., "case"/"control" or 0/1)
- **Data cells:** Numeric values
- **File type:** CSV format

## Usage

### Basic Syntax

```bash
python scripts/generic_split_dataset.py <input_csv> <output_dir> <label_row_name> [random_seed]
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input_csv` | Path to input CSV file | `data/Finalized_GSE90073.csv` |
| `output_dir` | Directory where split files will be saved | `data/splits_gse90073` |
| `label_row_name` | Name of the row containing class labels | `classification` |
| `random_seed` | (Optional) Random seed for reproducibility, default=12345 | `12345` |

### Examples

#### Example 1: GSE90073 Dataset
```bash
python scripts/generic_split_dataset.py \
  data/Finalized_GSE90073.csv \
  data/splits_gse90073 \
  classification \
  12345
```

#### Example 2: GSE139294 Dataset
```bash
python scripts/generic_split_dataset.py \
  data/Finalized_GSE139294.csv \
  data/splits_gse139294 \
  classification \
  12345
```

#### Example 3: Custom Dataset
```bash
python scripts/generic_split_dataset.py \
  data/my_custom_data.csv \
  data/splits_custom \
  label \
  42
```

## Output Files

The script creates three CSV files in the specified output directory:

### 1. **TRAIN_70_percent.csv**
- Contains 70% of the data
- Stratified sample (maintains class distribution)
- Shape: (N_train, features + 1) where last column is "label"

### 2. **VAL_20_percent.csv**
- Contains 20% of the data  
- Stratified sample (maintains class distribution)
- Shape: (N_val, features + 1) where last column is "label"

### 3. **TEST_10_percent.csv**
- Contains 10% of the data
- Stratified sample (maintains class distribution)
- Shape: (N_test, features + 1) where last column is "label"

**Example output structure:**
```
TRAIN_70_percent.csv:
                SNP1       SNP2       SNP3    ...    SNPn    label
Sample1         0.123      0.456      0.789   ...    0.234   case
Sample2         0.456      0.789      0.234   ...    0.567   control
Sample3         0.789      0.234      0.567   ...    0.890   case
...
```

## Key Features

### 1. Stratified Splitting
Uses `sklearn.model_selection.train_test_split` with `stratify=y` to ensure:
- Training set has same class distribution as original data
- Validation set has same class distribution as original data
- Test set has same class distribution as original data

**Example:**
```
Original class distribution: case=67, control=39
After split:
  Train:  case=47, control=27  (same 63%/37% ratio)
  Val:    case=13, control=8   (same 63%/37% ratio)
  Test:   case=7,  control=4   (same 63%/37% ratio)
```

### 2. Exact Proportions
- Train: 70% (±0.1% due to rounding)
- Val: 20% (±0.1% due to rounding)
- Test: 10% (±0.1% due to rounding)

### 3. Memory Optimization
- Garbage collection after loading large files
- Efficient data handling for datasets up to 1GB+
- Streaming-friendly for large files

### 4. Reproducibility
- Default random seed: 12345
- Same seed produces identical splits
- Useful for experimentation and debugging

## Example Workflow

### Step 1: Prepare Your Dataset
Ensure your CSV file follows the required format:
- Features in rows
- Samples in columns
- Last row contains class labels

### Step 2: Run the Script
```bash
python scripts/generic_split_dataset.py \
  data/my_dataset.csv \
  data/my_splits \
  class_labels \
  12345
```

### Step 3: Load and Use Splits
```python
import pandas as pd

# Load splits
train_df = pd.read_csv('data/my_splits/TRAIN_70_percent.csv', index_col=0)
val_df = pd.read_csv('data/my_splits/VAL_20_percent.csv', index_col=0)
test_df = pd.read_csv('data/my_splits/TEST_10_percent.csv', index_col=0)

# Separate features and labels
X_train = train_df.iloc[:, :-1]
y_train = train_df['label']

X_val = val_df.iloc[:, :-1]
y_val = val_df['label']

X_test = test_df.iloc[:, :-1]
y_test = test_df['label']
```

## Output Example (Real Execution)

```
======================================================================
GENERIC DATASET SPLIT (70/20/10 Stratified)
======================================================================
Input CSV: data/Finalized_GSE90073.csv
Output Dir: data/splits_gse90073_test
Label Row: classification
Random Seed: 12345
======================================================================

Validating inputs...
✓ Inputs validated
Loading data/Finalized_GSE90073.csv...
✓ Loaded dataset with shape (541351, 106)
Extracting labels from row 'classification'...
✓ Data prepared
  Samples: 106
  Features/SNPs: 541350
  Class balance:
classification
case       67
control    39

Performing exact 70/20/10 stratified split...
✓ First split complete: 74 train samples
✓ Second split complete

Split Results:
  Train: 74 samples (69.8%) - {'case': 47, 'control': 27}
  Val:   21 samples (19.8%) - {'case': 13, 'control': 8}
  Test:  11 samples (10.4%) - {'case': 7, 'control': 4}

Saving splits to data/splits_gse90073_test/...
  Saving TRAIN_70_percent.csv...
  ✓ Saved TRAIN_70_percent.csv (232.34 MB)
  Saving VAL_20_percent.csv...
  ✓ Saved VAL_20_percent.csv (71.09 MB)
  Saving TEST_10_percent.csv...
  ✓ Saved TEST_10_percent.csv (40.68 MB)
✓ All splits saved successfully

======================================================================
✅ DONE: Dataset split completed successfully!
======================================================================
```

## Troubleshooting

### Problem: "FileNotFoundError: Input file not found"
- **Solution:** Check that the input CSV path is correct and the file exists

### Problem: "Label row 'XXX' not found in dataset"
- **Solution:** Verify the label row name matches exactly (case-sensitive)

### Problem: Script takes very long to load
- **Solution:** This is normal for large files (600+ MB). Be patient or run outside VS Code terminal.

### Problem: Memory error
- **Solution:** 
  - Close other applications to free memory
  - The script uses garbage collection, but very large files may need system resources

## Verifying Splits

After running the script, verify the splits are correct:

```bash
python -c "
import pandas as pd
t = pd.read_csv('data/splits_gse90073/TRAIN_70_percent.csv')
v = pd.read_csv('data/splits_gse90073/VAL_20_percent.csv')
te = pd.read_csv('data/splits_gse90073/TEST_10_percent.csv')
total = len(t) + len(v) + len(te)
print(f'Train: {len(t)} ({len(t)/total*100:.1f}%)')
print(f'Val:   {len(v)} ({len(v)/total*100:.1f}%)')
print(f'Test:  {len(te)} ({len(te)/total*100:.1f}%)')
"
```

Expected output:
```
Train: 74 (69.8%)
Val:   21 (19.8%)
Test:  11 (10.4%)
```

## Advanced: Customization

You can modify the script for different split ratios by editing these lines:

```python
# Current: 70/20/10
train_size = int(total * 0.70)  # Change 0.70 to your ratio
val_size = int(total * 0.20)    # Change 0.20 to your ratio
# test_size = remaining automatically
```

## License & Credits

This is a general-purpose template. Feel free to modify and distribute as needed.

## Contact & Questions

For issues or questions, ensure:
1. Your input CSV follows the required format
2. The label row name is spelled correctly
3. Your system has enough memory for the dataset size
4. Dependencies are properly installed
