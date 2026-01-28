import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# PATHS
# ===============================
DATASET_PATH = "data/Finalized_GSE90073.csv"
OUTPUT_DIR = "data/splits_gse90073"
RANDOM_SEED = 12345

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATASET_PATH)
df = df.set_index("ID_REF")

# -------------------------------
# Separate LABEL row
# -------------------------------
labels = df.loc["classif"]        # Series: GSM → case/control
features = df.drop(index="classif")

# -------------------------------
# Transpose features
# -------------------------------
X = features.T                    # (samples × SNPs)
y = labels.loc[X.index]           # align labels

# Encode labels (DL needs numeric)
y = y.map({"control": 0, "case": 1})

print("Samples:", X.shape[0])
print("SNPs:", X.shape[1])
print("Class balance:\n", y.value_counts())

# ===============================
# 10% TEST SPLIT (STRATIFIED)
# ===============================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=RANDOM_SEED,
    stratify=y
)

# ===============================
# 70% TRAIN / 20% VAL
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.2222,  # 20% of total
    random_state=RANDOM_SEED,
    stratify=y_trainval
)

# ===============================
# SAVE CSV (label as LAST COLUMN)
# ===============================
def save_split(X, y, filename):
    df_out = X.copy()
    df_out["label"] = y
    df_out.to_csv(os.path.join(OUTPUT_DIR, filename))

save_split(X_train, y_train, "TRAIN_70_percent.csv")
save_split(X_val, y_val, "VAL_20_percent.csv")
save_split(X_test, y_test, "TEST_10_percent.csv")

print("DL-safe split completed")
