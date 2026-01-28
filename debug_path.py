import os

DATASET_PATH = r"C:\Users\Shadman Sakib\Downloads\SNP-Net\data\Finalized_GSE90073.csv"
print("DATASET_PATH:", DATASET_PATH)
print("Type:", type(DATASET_PATH))
print("Repr:", repr(DATASET_PATH))
print("File exists:", os.path.exists(DATASET_PATH))
print("Absolute path:", os.path.abspath(DATASET_PATH))
