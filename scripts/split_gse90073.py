print("SCRIPT STARTED: GSE90073 Dataset Splitting")
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# ===============================
# PATHS
# ===============================
DATASET_PATH = "data/Finalized_GSE90073.csv"
OUTPUT_DIR = "data/splits_gse90073"
RANDOM_SEED = 12345

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# ===============================
# LOAD DATA
# ===============================
print(f"Loading {DATASET_PATH}...", flush=True)
sys.stdout.flush()
df = pd.read_csv(DATASET_PATH, index_col=0, low_memory=False)
print(f"✓ Loaded dataset with shape {df.shape}", flush=True)

# ===============================
# Separate LABEL row
# ===============================
print("Extracting labels...", flush=True)
labels = df.loc["classification"]        # Series: GSM → case/control
features = df.drop(index="classification")

# ===============================
# Transpose features
# ===============================
X = features.T                    # (samples × SNPs)
y = labels.loc[X.index]           # align labels

print(f"Samples: {X.shape[0]}")
print(f"SNPs: {X.shape[1]}")
print(f"Class balance:\n{y.value_counts()}", flush=True)

# ===============================
# EXACT 70% TRAIN / 20% VAL / 10% TEST SPLIT (STRATIFIED)
# ===============================
print("\nPerforming exact 70/20/10 stratified split...", flush=True)
total = len(X)
train_size = int(total * 0.70)
val_size = int(total * 0.20)
test_size = total - train_size - val_size

# First split: 70% train, 30% temp (which will be val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    train_size=train_size,
    random_state=RANDOM_SEED,
    stratify=y
)

# Second split: split temp into 20% val (67% of temp) and 10% test (33% of temp)
val_ratio = val_size / len(X_temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    train_size=val_ratio,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print(f"Train: {len(X_train)} samples - Class distribution: {y_train.value_counts().to_dict()}")
print(f"Val:   {len(X_val)} samples - Class distribution: {y_val.value_counts().to_dict()}")
print(f"Test:  {len(X_test)} samples - Class distribution: {y_test.value_counts().to_dict()}", flush=True)

# ===============================
# SAVE CSV (label as LAST COLUMN)
# ===============================
def save_split(X, y, filename):
    df_out = X.copy()
    df_out["label"] = y
    filepath = os.path.join(OUTPUT_DIR, filename)
    df_out.to_csv(filepath)
    print(f"✓ Saved {filepath}", flush=True)

print("\nSaving splits...", flush=True)
save_split(X_train, y_train, "TRAIN_70_percent.csv")
save_split(X_val, y_val, "VAL_20_percent.csv")
save_split(X_test, y_test, "TEST_10_percent.csv")

print("\n✅ DONE: GSE90073 split completed successfully", flush=True)
