#!/usr/bin/env python3
"""
Generic Dataset Splitter - Reusable template for 70/20/10 stratified split.
Suitable for any dataset with the format: features as rows, samples as columns, last row as labels.

Usage:
    python generic_split_dataset.py <input_csv> <output_dir> <label_row_name> [--seed 12345]

Example:
    python generic_split_dataset.py data/Finalized_GSE90073.csv data/splits classification 12345
"""

import sys
import os
import argparse
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def validate_inputs(input_file, output_dir, label_row):
    """Validate that input file exists and contains the label row."""
    if not os.path.isfile(input_file):
        print("[ERROR] Input file does not exist:", input_file)
        return False
    
    # Check if label row exists in the file
    try:
        df = pd.read_csv(input_file, index_col=0, low_memory=False)
        if label_row not in df.index:
            print("[ERROR] Label row '{0}' not found in CSV".format(label_row))
            return False
    except Exception as e:
        print("[ERROR] Failed to read input file: {0}".format(str(e)))
        return False
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    return True


def load_and_prepare_data(input_file, label_row):
    """Load data and prepare features + labels (transpose to samples x features format)."""
    print("[INFO] Loading dataset from:", input_file)
    
    # Read CSV (features as rows, samples as columns)
    df = pd.read_csv(input_file, index_col=0, low_memory=False)
    
    # Extract labels (last row of data)
    if label_row not in df.index:
        print("[ERROR] Label row '{0}' not found".format(label_row))
        return None, None, None
    
    labels = df.loc[label_row]
    features = df.drop(label_row)
    
    # Transpose to get samples as rows
    X = features.T
    # Convert labels to numeric codes
    unique_labels = labels.unique()
    label_to_code = {label: idx for idx, label in enumerate(unique_labels)}
    y = labels.map(label_to_code).astype(int)
    y_names = labels
    
    print("[INFO] Dataset shape: {0} samples, {1} features".format(X.shape[0], X.shape[1]))
    class_counts = {}
    for label in unique_labels:
        class_counts[label] = (labels == label).sum()
    print("[INFO] Classes: {0}".format(class_counts))
    
    del df, labels, features
    gc.collect()
    
    return X, y, y_names


def perform_stratified_split(X, y, train_size=0.70, val_size=0.20, test_size=0.10, random_seed=12345):
    """Perform stratified train/val/test split."""
    total = len(X)
    
    # Phase 1: Split train (70%) from temp (30%)
    train_size_samples = int(total * train_size)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_size_samples, 
        random_state=random_seed, 
        stratify=y
    )
    
    # Phase 2: Split temp (30%) into val (20% of total) and test (10% of total)
    val_size_samples = int(total * val_size)
    val_ratio = val_size_samples / len(X_temp)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=random_seed,
        stratify=y_temp
    )
    
    del X, y, X_temp, y_temp
    gc.collect()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir, dataset_name="split"):
    """Save split datasets to CSV files."""
    train_file = os.path.join(output_dir, "{0}_train.csv".format(dataset_name))
    val_file = os.path.join(output_dir, "{0}_val.csv".format(dataset_name))
    test_file = os.path.join(output_dir, "{0}_test.csv".format(dataset_name))
    
    try:
        # Add labels as last column
        train_df = X_train.copy()
        train_df['label'] = y_train.values
        train_df.to_csv(train_file, index=False)
        print("[OK] Saved training split ({0} samples) to: {1}".format(len(train_df), train_file))
        
        val_df = X_val.copy()
        val_df['label'] = y_val.values
        val_df.to_csv(val_file, index=False)
        print("[OK] Saved validation split ({0} samples) to: {1}".format(len(val_df), val_file))
        
        test_df = X_test.copy()
        test_df['label'] = y_test.values
        test_df.to_csv(test_file, index=False)
        print("[OK] Saved test split ({0} samples) to: {1}".format(len(test_df), test_file))
        
        return True
    except Exception as e:
        print("[ERROR] Failed to save splits: {0}".format(str(e)))
        return False


def print_summary(y_train, y_val, y_test):
    """Print split summary statistics."""
    total = len(y_train) + len(y_val) + len(y_test)
    train_pct = (len(y_train) / total) * 100
    val_pct = (len(y_val) / total) * 100
    test_pct = (len(y_test) / total) * 100
    
    print("\n" + "="*60)
    print("STRATIFIED SPLIT SUMMARY")
    print("="*60)
    print("Total samples: {0}".format(total))
    print("Training:   {0} samples ({1:.1f}%)".format(len(y_train), train_pct))
    print("Validation: {0} samples ({1:.1f}%)".format(len(y_val), val_pct))
    print("Test:       {0} samples ({1:.1f}%)".format(len(y_test), test_pct))
    
    print("\nClass distribution (stratified across splits):")
    print("Training:   ", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Validation: ", dict(zip(*np.unique(y_val, return_counts=True))))
    print("Test:       ", dict(zip(*np.unique(y_test, return_counts=True))))
    print("="*60 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generic dataset splitter (70/20/10 stratified split)")
    parser.add_argument("input_csv", help="Path to input CSV file (features as rows, samples as columns, last row as labels)")
    parser.add_argument("output_dir", help="Directory to save split files")
    parser.add_argument("label_row", help="Name of the row containing labels")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility (default: 12345)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_inputs(args.input_csv, args.output_dir, args.label_row):
        print("[ERROR] Input validation failed")
        sys.exit(1)
    
    # Load and prepare data
    X, y, y_names = load_and_prepare_data(args.input_csv, args.label_row)
    if X is None:
        print("[ERROR] Data loading failed")
        sys.exit(1)
    
    # Perform stratified split
    print("[INFO] Performing stratified 70/20/10 split with seed {0}".format(args.seed))
    X_train, X_val, X_test, y_train, y_val, y_test = perform_stratified_split(
        X, y, random_seed=args.seed
    )
    
    # Save splits
    dataset_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    if not save_splits(X_train, X_val, X_test, y_train, y_val, y_test, args.output_dir, dataset_name):
        print("[ERROR] Failed to save splits")
        sys.exit(1)
    
    # Print summary
    print_summary(y_train, y_val, y_test)
    
    print("[SUCCESS] Stratified split completed successfully!")


if __name__ == "__main__":
    main()
