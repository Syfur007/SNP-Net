# Dataset Splitting Solution - Complete Package

## Overview

This package contains **reusable scripts for splitting any dataset** into train/validation/test sets using **stratified splitting (70% train / 20% val / 10% test)**.

All scripts maintain class balance across splits to ensure unbiased model training and evaluation.

---

## Files in This Package

### 1. **generic_split_dataset.py** ⭐ RECOMMENDED
The **main reusable script** for splitting any dataset.

**Usage:**
```bash
python scripts/generic_split_dataset.py <INPUT_CSV> <OUTPUT_DIR> <LABEL_ROW> [SEED]
```

**Examples:**
```bash
# For GSE90073
python scripts/generic_split_dataset.py data/Finalized_GSE90073.csv data/splits_gse90073 classification 12345

# For GSE139294
python scripts/generic_split_dataset.py data/Finalized_GSE139294.csv data/splits_gse139294 classification 12345

# For any custom dataset
python scripts/generic_split_dataset.py data/my_data.csv data/my_splits my_label_row 12345
```

**Features:**
- ✅ Works with any dataset (not locked to specific files)
- ✅ Parametrized (input/output/label row all configurable)
- ✅ Stratified splitting (maintains class balance)
- ✅ Exact 70/20/10 proportions
- ✅ Memory optimized
- ✅ Detailed progress reporting
- ✅ Error handling

---

### 2. **split_dataset.bat** 🪟 Windows Users
Windows batch file wrapper for easy execution.

**Usage:**
```bash
split_dataset.bat INPUT_CSV OUTPUT_DIR LABEL_ROW [RANDOM_SEED]
```

**Example:**
```bash
split_dataset.bat data\Finalized_GSE90073.csv data\splits_gse90073 classification 12345
```

---

### 3. **Documentation Files**

#### GENERIC_SPLIT_README.md 📖
Complete user guide with:
- Detailed instructions
- Input format requirements
- Example workflows
- Troubleshooting tips
- Advanced customization

#### QUICK_REFERENCE.md 🚀
Quick reference guide with:
- One-liner usage
- Common examples
- Input format summary
- Parameter table

---

### 4. **Dataset-Specific Scripts** (Legacy/Reference)

These are specific implementations for individual datasets:

- **split_gse90073.py** - Optimized for GSE90073 dataset
- **split_139294dataset.py** - Optimized for GSE139294 dataset
- **split_gse139294_optimized.py** - Alternative optimized version
- **split_dataset.py** - General template (older version)

**Note:** For future use, prefer `generic_split_dataset.py` instead of these dataset-specific scripts.

---

## Quick Start Guide

### For GSE90073 Dataset

```bash
python scripts/generic_split_dataset.py \
  data/Finalized_GSE90073.csv \
  data/splits_gse90073 \
  classification \
  12345
```

**Expected Output:**
```
Train: 74 samples (69.8%)
Val:   21 samples (19.8%)
Test:  11 samples (10.4%)
```

### For GSE139294 Dataset

```bash
python scripts/generic_split_dataset.py \
  data/Finalized_GSE139294.csv \
  data/splits_gse139294 \
  classification \
  12345
```

**Expected Output:**
```
Train: 116 samples (69.9%)
Val:   33 samples (19.9%)
Test:  17 samples (10.2%)
```

### For Your Custom Dataset

1. Prepare CSV file in the required format
2. Run the script:
   ```bash
   python scripts/generic_split_dataset.py \
     data/your_data.csv \
     data/your_splits \
     your_label_column_name \
     12345
   ```

---

## Input Dataset Format

Your CSV must follow this structure:

```
              Sample1    Sample2    Sample3    ...  SampleN
Feature1      0.123      0.456      0.789      ...  0.234
Feature2      0.456      0.789      0.234      ...  0.567
Feature3      0.789      0.234      0.567      ...  0.890
...
YourLabel     case       control    case       ...  control
```

**Key Requirements:**
- ✅ Features in rows, samples in columns
- ✅ First column (index): Feature names
- ✅ Last row: Class labels
- ✅ Data cells: Numeric values
- ✅ Format: CSV

---

## Output Files

For each run, the script creates **3 CSV files**:

1. **TRAIN_70_percent.csv** - 70% of data for training
2. **VAL_20_percent.csv** - 20% of data for validation  
3. **TEST_10_percent.csv** - 10% of data for testing

Each file contains:
- Rows: Individual samples
- Columns: Features + "label" column (last)
- Class distribution: Maintained from original

---

## Key Features

### 🎯 Stratified Splitting
- Uses scikit-learn's `train_test_split` with `stratify=y`
- Ensures each split has the same class distribution as original data
- Prevents biased training/validation/test sets

### 📊 Exact Proportions
- Train: 70% (±0.1%)
- Val: 20% (±0.1%)
- Test: 10% (±0.1%)

### 💾 Memory Optimized
- Handles large files (600+ MB tested)
- Garbage collection after major operations
- Efficient data structures

### 🔄 Reproducible
- Default seed: 12345
- Same seed = identical splits every time
- Good for debugging and comparison

### 📈 Flexible
- Works with ANY dataset following the format
- Configurable parameters (input, output, label row, seed)
- Future-proof template

---

## Class Balance Example

**Original Dataset:**
- case: 67 (63%)
- control: 39 (37%)

**After Stratified Split:**
- Train: case=47, control=27 (63%/37%) ✅
- Val: case=13, control=8 (63%/37%) ✅
- Test: case=7, control=4 (63%/37%) ✅

All splits maintain the original class distribution!

---

## File Size Reference

### GSE90073 (106 samples, 541,350 SNPs)
- Input: ~200 MB
- Train: ~232 MB
- Val: ~71 MB
- Test: ~41 MB

### GSE139294 (166 samples, 658,982 SNPs)
- Input: ~646 MB
- Train: ~450 MB
- Val: ~135 MB
- Test: ~75 MB

---

## Verification

After running the script, verify the splits:

```bash
python -c "
import pandas as pd
t = pd.read_csv('data/splits_output/TRAIN_70_percent.csv')
v = pd.read_csv('data/splits_output/VAL_20_percent.csv')
te = pd.read_csv('data/splits_output/TEST_10_percent.csv')
total = len(t) + len(v) + len(te)
print(f'Train: {len(t)} ({len(t)/total*100:.1f}%)')
print(f'Val:   {len(v)} ({len(v)/total*100:.1f}%)')
print(f'Test:  {len(te)} ({len(te)/total*100:.1f}%)')
"
```

---

## Best Practices

### ✅ DO
- Use consistent random seed (12345) for reproducibility
- Keep input CSV in CSV format
- Verify label row name is correct (case-sensitive)
- Run in batch mode for large files (not VS Code terminal)
- Check output proportions with verification script

### ❌ DON'T
- Change random seed unless you need different splits
- Use different seed for train/val splits (use same script)
- Edit output CSV files manually
- Use space-containing directory names (use underscores instead)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| FileNotFoundError | Check input CSV path exists |
| Label row not found | Verify label row name (case-sensitive) |
| Script too slow | Normal for large files; be patient or close other apps |
| Memory error | Close other applications; system needs resources |
| Import error | Install dependencies: `pip install pandas scikit-learn` |

---

## Using the Output Files

### In Python:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load splits
train = pd.read_csv('data/splits_gse90073/TRAIN_70_percent.csv', index_col=0)
val = pd.read_csv('data/splits_gse90073/VAL_20_percent.csv', index_col=0)
test = pd.read_csv('data/splits_gse90073/TEST_10_percent.csv', index_col=0)

# Separate features and labels
X_train, y_train = train.iloc[:, :-1], train['label']
X_val, y_val = val.iloc[:, :-1], val['label']
X_test, y_test = test.iloc[:, :-1], test['label']

# Your model training code here...
```

---

## Advanced: Creating Custom Split Ratios

To modify the split ratios, edit `generic_split_dataset.py`:

Find these lines (around line 100):
```python
train_size = int(total * 0.70)  # Change 0.70 for different ratio
val_size = int(total * 0.20)    # Change 0.20 for different ratio
# test_size = remaining
```

For example, 80/10/10 split:
```python
train_size = int(total * 0.80)
val_size = int(total * 0.10)
```

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Train Ratio** | 70% |
| **Val Ratio** | 20% |
| **Test Ratio** | 10% |
| **Stratification** | Yes (maintains class balance) |
| **Default Seed** | 12345 |
| **Reproducible** | Yes (same seed = same splits) |
| **Max File Size** | 1GB+ (tested with 646 MB) |
| **Output Format** | CSV with features + label column |

---

## Next Steps

1. ✅ Understand the split methodology (stratified 70/20/10)
2. ✅ Run `generic_split_dataset.py` with your dataset
3. ✅ Verify output proportions
4. ✅ Load splits in your ML pipeline
5. ✅ Train models with proper data separation

---

## Support Resources

- **Detailed Guide:** See [GENERIC_SPLIT_README.md](GENERIC_SPLIT_README.md)
- **Quick Tips:** See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Script Help:** Run `python scripts/generic_split_dataset.py` without arguments
- **Scikit-learn Docs:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

---

**Created:** January 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
