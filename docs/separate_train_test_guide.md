# Separate Train/Test CSV Files Guide

## Overview

The framework **automatically creates and manages separate train/test CSV files**. If separate files don't exist, the framework will automatically split a single data file and save them.

## Key Features

✅ **Auto-Detection**: Framework automatically detects `*train*.csv` and `*test*.csv` files  
✅ **Auto-Splitting**: If separate files don't exist, framework splits the single file and saves them  
✅ **K-Fold Support**: K-fold cross-validation runs on train file only; test file used as final test set  
✅ **No Data Leakage**: Normalization and feature selection computed on train set only  
✅ **Shared Label Config**: Label handling (labels in last row, label_row_name, label_file) applies to both files  

## How It Works

### Complete Data Flow:

1. **Initialization**: DataModule is created with `data_file` configuration

2. **Data Preparation Phase** (`prepare_data()`):
   - Scans data directory for existing `*train*.csv` and `*test*.csv` files
   - **If found**: Uses them directly
   - **If NOT found**: Automatically splits the single `data_file` using `train_val_test_split[2]` (test ratio)
     - Loads single file
     - Randomly splits into train/test
     - Saves as separate CSV files in the same directory
     - Proceeds with those files

3. **Setup Phase** (`setup()`):
   - Train file is split into train/val (or k-folds)
   - Test file is used **directly as final test set** (NOT split)

4. **Preprocessing**:
   - Normalization statistics (mean/std) computed on train set only
   - Feature selection performed on train set only
   - Both computed statistics applied to val and test sets

## Configuration Examples

### Example 1: Auto-Detect or Auto-Split (Recommended)

**YAML Config** (`configs/data/example.yaml`):
```yaml
_target_: src.data.datamodule.DataModule

# Single file path - framework will auto-detect or split this
data_file: "data/dataset.csv"

# Label configuration (applies to both train and test files)
labels_in_last_row: true
label_file: null
label_row_name: null

# Train/test split ratio (used when splitting single file)
train_val_test_split: [0.0, 0.0, 0.2]  # [train, val, test] - test=0.2 means 80/20 split

# K-fold configuration (optional)
num_folds: 5
current_fold: 0

# Other settings
batch_size: 32
normalize: true
has_header: true
has_index: true
feature_selection: ...  # optional
```

**What happens**:
```
First run:
├── Checks for dataset_train.csv and dataset_test.csv → Not found
├── Loads dataset.csv
├── Splits into 80% train, 20% test
├── Saves as dataset_train.csv and dataset_test.csv
└── Uses separate files for training

Subsequent runs:
├── Checks for dataset_train.csv and dataset_test.csv → Found!
└── Uses existing separate files directly
```

### Example 2: With K-Fold Cross-Validation

**YAML Config**:
```yaml
_target_: src.data.datamodule.DataModule

data_file: "data/cancer_dataset.csv"
labels_in_last_row: true

# Test ratio for splitting (train gets rest)
train_val_test_split: [0.0, 0.0, 0.2]  # 80% train, 20% test

# K-fold on train file only
num_folds: 5
current_fold: 0

batch_size: 32
normalize: true
```

**Training all 5 folds**:
```bash
# Automatically splits single file into cancer_dataset_train.csv and cancer_dataset_test.csv
# Then runs k-fold on train file, keeping test file fixed
python src/train.py data=cancer_dataset
```

### Example 3: Already Have Separate Files

**Data Directory**:
```
data/
├── autism_train.csv   # Pre-split train data
└── autism_test.csv    # Pre-split test data
```

**YAML Config**:
```yaml
_target_: src.data.datamodule.DataModule

# Point to either file - framework will auto-detect both
data_file: "data/autism.csv"  # Doesn't need to exist

labels_in_last_row: true
train_val_test_split: [0.8, 0.2, 0.0]  # Ignored, files already split

batch_size: 32
normalize: true
```

**What happens**:
```
├── Checks for autism_train.csv and autism_test.csv → Found!
├── Uses them directly (no splitting needed)
└── Proceeds with training
```

## File Naming Convention

The framework auto-detects files matching:
- **Train files**: `*train*.csv` pattern (e.g., `dataset_train.csv`, `train_data.csv`, `cancer_TRAIN.csv`)
- **Test files**: `*test*.csv` pattern (e.g., `dataset_test.csv`, `test_data.csv`, `cancer_TEST.csv`)

When splitting, the framework creates files named:
- Train: `{original_basename}_train.csv`
- Test: `{original_basename}_test.csv`

Examples:
```
dataset.csv        → dataset_train.csv + dataset_test.csv
cancer_data.csv    → cancer_data_train.csv + cancer_data_test.csv
snp_complete.csv   → snp_complete_train.csv + snp_complete_test.csv
```

## DataModule Parameters

### Relevant Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_file` | str | `"data/snp_data.csv"` | Path to single data file. Can be auto-split if separate files don't exist. |
| `train_val_test_split` | tuple | `(0.7, 0.15, 0.15)` | `[train, val, test]` ratios. When splitting single file, only `[2]` (test ratio) is used. |
| `num_folds` | int | `None` | If > 1, applies k-fold to train file only. Test file remains fixed. |
| `labels_in_last_row` | bool | `True` | Whether labels are in the last row. Applied to both train and test files. |
| `label_file` | str | `None` | Optional separate label file. Applied to both train and test files. |
| `label_row_name` | str | `None` | Row name containing labels. Applied to both train and test files. |
| `auto_detect_train_test` | bool | `True` | If True, auto-detect or auto-split train/test files. Defaults to True. |

## Implementation Details

### Auto-Detection & Splitting Flow

```
prepare_data() [called once before training]
├── If separate files detected
│  └── Use them directly
└── Else if single file exists
   ├── Load single file
   ├── Split by train_val_test_split[2] ratio
   ├── Save as {basename}_train.csv and {basename}_test.csv
   └── Use the saved files
```

### Key Methods

- **`_split_if_needed(log)`**: Checks for existing separate files, or triggers splitting
- **`_split_single_file_into_train_test(log)`**: Loads, splits, and saves separate files

## Common Scenarios

### Scenario A: First Time Training with Single File

```bash
# Initial config
python src/train.py data=new_dataset

# What happens:
# 1. prepare_data() called
# 2. Checks for new_dataset_train.csv and new_dataset_test.csv → Not found
# 3. Loads new_dataset.csv
# 4. Splits (default 80/20 based on train_val_test_split[2])
# 5. Saves new_dataset_train.csv and new_dataset_test.csv
# 6. Training proceeds with split files
```

### Scenario B: Re-running or Multi-Fold Training

```bash
# Second run with same dataset
python src/train.py data=new_dataset

# What happens:
# 1. prepare_data() called
# 2. Checks for new_dataset_train.csv and new_dataset_test.csv → Found!
# 3. Uses existing split files directly
# 4. Training proceeds
```

### Scenario C: Custom Test Ratio

```yaml
# configs/data/custom_split.yaml
data_file: "data/dataset.csv"
train_val_test_split: [0.0, 0.0, 0.3]  # 70% train, 30% test
```

```bash
python src/train.py data=custom_split

# Creates:
# - dataset_train.csv (70% of samples)
# - dataset_test.csv (30% of samples)
```

## Important Notes

- **Deterministic Splitting**: Uses `random_state=42` for reproducible splits
- **Sample-wise Split**: Samples (columns in the original CSV) are randomly split, not rows
- **Saved Files**: Once split files are created, they persist. Rerunning won't re-split unless you delete them
- **Label Consistency**: Both train and test files must have labels in the same location
- **Backward Compatible**: Fully compatible with single-file configs from previous versions
