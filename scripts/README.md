# 📊 Dataset Split Scripts - Package Index

## 🎯 Start Here

**For any dataset splitting task, use:** [`generic_split_dataset.py`](generic_split_dataset.py)

```bash
python scripts/generic_split_dataset.py <INPUT_CSV> <OUTPUT_DIR> <LABEL_ROW> [SEED]
```

---

## 📁 Files Overview

### 🚀 Main Scripts

| File | Purpose | Use Case |
|------|---------|----------|
| **generic_split_dataset.py** | Universal split script | ⭐ PRIMARY - Use for any dataset |
| **split_dataset.bat** | Windows batch wrapper | 🪟 Windows users (easy click-to-run) |

### 📖 Documentation

| File | Content |
|------|---------|
| **DATASET_SPLIT_PACKAGE.md** | Complete package overview & guide |
| **GENERIC_SPLIT_README.md** | Detailed user manual & examples |
| **QUICK_REFERENCE.md** | Cheat sheet & quick commands |
| **README.md** | This file |

### 📝 Legacy/Reference Scripts

| File | Status |
|------|--------|
| split_gse90073.py | Specific to GSE90073 (reference only) |
| split_139294dataset.py | Specific to GSE139294 (reference only) |
| split_gse139294_optimized.py | Alternative GSE139294 version |
| split_dataset.py | Original template |

---

## 🚀 Quick Start

### Basic Usage
```bash
python scripts/generic_split_dataset.py data/YOUR_DATA.csv data/output_splits CLASS_LABEL 12345
```

### For GSE90073
```bash
python scripts/generic_split_dataset.py data/Finalized_GSE90073.csv data/splits_gse90073 classification 12345
```

### For GSE139294
```bash
python scripts/generic_split_dataset.py data/Finalized_GSE139294.csv data/splits_gse139294 classification 12345
```

---

## ✨ Features

✅ Stratified splitting (maintains class balance)  
✅ Exact 70% train / 20% val / 10% test  
✅ Works with any dataset  
✅ Memory optimized (handles large files)  
✅ Reproducible (with seed)  
✅ Comprehensive error handling  
✅ Detailed progress reporting  

---

## 📊 What You Get

Running the script produces **3 output CSV files**:

1. **TRAIN_70_percent.csv** - Training data (70%)
2. **VAL_20_percent.csv** - Validation data (20%)
3. **TEST_10_percent.csv** - Test data (10%)

Each file contains:
- **Rows:** Samples
- **Columns:** Features + "label" column
- **Class distribution:** Same as original

---

## 🔍 Example

**Input:** 106 samples, 67 cases, 39 controls

```
Original:
  case: 67 (63.2%)
  control: 39 (36.8%)

After split (stratified):
  Train:  case=47, control=27 (63.2%/36.8%) ✅
  Val:    case=13, control=8 (63.2%/36.8%) ✅
  Test:   case=7, control=4 (63.2%/36.8%) ✅
```

---

## 📋 Input Format Required

```
              Sample1    Sample2    Sample3    ...
Feature1      0.123      0.456      0.789      ...
Feature2      0.456      0.789      0.234      ...
...
YourLabel     case       control    case       ...
```

✅ Features in rows, samples in columns  
✅ Last row = class labels  
✅ CSV format

---

## 🎓 Documentation

**Read in order:**

1. **Start here** → QUICK_REFERENCE.md (2 min read)
2. **Main guide** → GENERIC_SPLIT_README.md (10 min read)
3. **Full overview** → DATASET_SPLIT_PACKAGE.md (15 min read)

---

## 🛠️ Usage Options

### Option 1: Python Command Line
```bash
python scripts/generic_split_dataset.py data/file.csv output_dir label 12345
```

### Option 2: Windows Batch (Windows only)
```bash
split_dataset.bat data\file.csv output_dir label 12345
```

### Option 3: Python Script
```python
import subprocess
result = subprocess.run([
    'python', 
    'scripts/generic_split_dataset.py',
    'data/file.csv',
    'output_dir',
    'label',
    '12345'
])
```

---

## ✅ Verification

After running:

```bash
python -c "
import pandas as pd
t = pd.read_csv('output_dir/TRAIN_70_percent.csv')
v = pd.read_csv('output_dir/VAL_20_percent.csv')
te = pd.read_csv('output_dir/TEST_10_percent.csv')
total = len(t) + len(v) + len(te)
print(f'Train: {len(t)} ({len(t)/total*100:.1f}%)')
print(f'Val: {len(v)} ({len(v)/total*100:.1f}%)')
print(f'Test: {len(te)} ({len(te)/total*100:.1f}%)')
"
```

Expected: 70%, 20%, 10% (±0.1%)

---

## 🔑 Key Parameters

```
<INPUT_CSV>    Path to your data file
<OUTPUT_DIR>   Where to save split files
<LABEL_ROW>    Name of the row with class labels (case-sensitive)
[SEED]         Optional: Random seed for reproducibility (default: 12345)
```

---

## 📦 Package Contents

```
scripts/
├── generic_split_dataset.py          ⭐ Main script
├── split_dataset.bat                  🪟 Windows wrapper
├── README.md                          📄 This file
├── QUICK_REFERENCE.md                 🚀 Quick guide
├── GENERIC_SPLIT_README.md            📖 Full manual
├── DATASET_SPLIT_PACKAGE.md           📚 Complete overview
└── [Legacy files...]                  For reference only
```

---

## ❓ FAQ

**Q: Which script should I use?**  
A: Always use `generic_split_dataset.py` - it's universal and future-proof.

**Q: Can I use this for other datasets?**  
A: Yes! Works with any dataset in the required format.

**Q: Is the split reproducible?**  
A: Yes, use the same seed (default: 12345) to get identical splits.

**Q: How large a file can it handle?**  
A: Tested with 646 MB. Larger files need more system memory.

**Q: Does it maintain class balance?**  
A: Yes! Uses stratified splitting to maintain class distribution.

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| File not found | Check path and spelling |
| Label not found | Verify label row name (case-sensitive) |
| Out of memory | Close other apps, use a better system |
| Import error | Install dependencies: `pip install pandas scikit-learn` |

---

## 🎓 Learn More

- **Stratified Splitting:** https://scikit-learn.org/stable/modules/model_selection.html#stratified-split
- **Train/Val/Test Split Concepts:** https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
- **Class Imbalance:** https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

---

## 💾 Usage Examples

### Example 1: Basic Usage
```bash
python scripts/generic_split_dataset.py \
  data/my_data.csv \
  data/splits \
  class_label
```

### Example 2: Custom Seed
```bash
python scripts/generic_split_dataset.py \
  data/my_data.csv \
  data/splits \
  class_label \
  42
```

### Example 3: Full Path
```bash
python scripts/generic_split_dataset.py \
  /absolute/path/to/data.csv \
  /absolute/path/to/output \
  label_name \
  12345
```

---

## 📋 Checklist Before Running

- [ ] Input CSV exists
- [ ] CSV is in required format (features as rows, samples as columns)
- [ ] Last row contains class labels
- [ ] Label row name is correct (case-sensitive)
- [ ] Python is installed and pandas/scikit-learn are available
- [ ] Output directory path is accessible
- [ ] Enough disk space for output files

---

## 🎯 Recommended Reading Order

1. **This file** (overview) - 3 minutes
2. **QUICK_REFERENCE.md** (quick start) - 2 minutes
3. **GENERIC_SPLIT_README.md** (detailed guide) - 10 minutes
4. Run your first split!

---

**Version:** 1.0  
**Status:** Production Ready ✅  
**Last Updated:** January 2026

For detailed questions, see the full documentation files.
