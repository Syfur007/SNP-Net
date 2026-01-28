# Quick Reference - Generic Split Script

## One-Liner Usage

```bash
python scripts/generic_split_dataset.py <INPUT_CSV> <OUTPUT_DIR> <LABEL_ROW> [SEED]
```

## Common Use Cases

### GSE90073
```bash
python scripts/generic_split_dataset.py data/Finalized_GSE90073.csv data/splits_gse90073 classification 12345
```

### GSE139294  
```bash
python scripts/generic_split_dataset.py data/Finalized_GSE139294.csv data/splits_gse139294 classification 12345
```

### New Dataset
```bash
python scripts/generic_split_dataset.py data/my_data.csv data/my_splits_output label_column_name 12345
```

## What It Does

1. ✅ Loads your CSV file
2. ✅ Separates features from labels (last row)
3. ✅ Creates stratified 70/20/10 split
4. ✅ Saves 3 CSV files:
   - `TRAIN_70_percent.csv` (70%)
   - `VAL_20_percent.csv` (20%)
   - `TEST_10_percent.csv` (10%)

## Output Files

Each output file has:
- Rows: Samples
- Columns: Features + "label" column
- Structure: Same as input but split and labeled

## Verification

```bash
python -c "
import pandas as pd
t=pd.read_csv('data/splits_output/TRAIN_70_percent.csv')
v=pd.read_csv('data/splits_output/VAL_20_percent.csv')
te=pd.read_csv('data/splits_output/TEST_10_percent.csv')
print(f'Train: {len(t)}, Val: {len(v)}, Test: {len(te)}')"
```

## Key Features

- 🎯 Stratified splitting (maintains class balance)
- 📊 Exact 70/20/10 proportions
- 💾 Memory optimized (handles large files)
- 🔄 Reproducible (with seed)
- 📈 Works with any dataset format

## Input Format Required

```
              Sample1  Sample2  Sample3  ...  SampleN
Feature1      0.123    0.456    0.789    ...  0.234
Feature2      0.456    0.789    0.234    ...  0.567
Feature3      0.789    0.234    0.567    ...  0.890
...
classification case    control   case    ...  control
```

## Parameters

| Param | Meaning | Example |
|-------|---------|---------|
| INPUT_CSV | Data file | `data/dataset.csv` |
| OUTPUT_DIR | Save location | `data/splits` |
| LABEL_ROW | Last row name | `classification` |
| SEED | Random seed (optional) | `12345` |

---

**For detailed guide, see:** [GENERIC_SPLIT_README.md](GENERIC_SPLIT_README.md)
