from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split, Subset
from sklearn.model_selection import KFold

from src.data.components.dataset import Dataset


class DataModule(LightningDataModule):
    """`LightningDataModule` for SNP (Single Nucleotide Polymorphism) dataset.

    This datamodule handles CSV files where:
    - Samples are stored as columns
    - Features (SNPs) are stored as rows
    - Labels (case/control) are in the last row of the CSV file by default
    - Alternatively, labels can be in a separate CSV file or a named row

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_file: str = "data/snp_data.csv",
        label_file: Optional[str] = None,
        label_row_name: Optional[str] = None,
        labels_in_last_row: bool = True,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize: bool = True,
        has_header: bool = True,
        has_index: bool = True,
        # K-Fold parameters (optional)
        num_folds: Optional[int] = None,
        current_fold: int = 0,
        # Feature selection parameters (optional)
        feature_selection: Optional[Dict[str, Any]] = None,
        # Data directory option
        data_dir: Optional[str] = None,
        # Separate split files (optional - if provided, overrides train_val_test_split)
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
    ) -> None:
        """Initialize a `SNPDataModule`.

        :param data_file: Path to the CSV file containing SNP data (samples as columns, SNPs as rows).
        :param label_file: Optional path to CSV file containing labels. If None, labels must be in data_file.
        :param label_row_name: Name of the row in data_file containing labels (if labels are in data_file).
        :param labels_in_last_row: Whether labels are in the last row of the CSV file. Defaults to True.
        :param train_val_test_split: Proportions for train, validation and test split. Defaults to (0.7, 0.15, 0.15).
        :param batch_size: The batch size. Defaults to 32.
        :param num_workers: The number of workers. Defaults to 0.
        :param pin_memory: Whether to pin memory. Defaults to False.
        :param normalize: Whether to normalize the data (z-score normalization). Defaults to True.
        :param has_header: Whether the CSV file has a header row (sample names). Defaults to True.
        :param has_index: Whether the CSV file has an index column (SNP names). Defaults to True.
        :param num_folds: Number of folds for k-fold cross validation. If None, uses regular train/val/test split. Defaults to None.
        :param current_fold: Current fold index for k-fold CV (0 to num_folds-1). Defaults to 0.
        :param feature_selection: Optional feature selection configuration dict with keys:
            - method: str (amgm, cosine, variance, mutual_info, l1)
            - k: int (number of features to select)
            - Additional method-specific parameters (threshold, mode, C, etc.)
        :param data_dir: Optional root directory for data files. If provided, data_file and label_file paths are resolved relative to this directory.
        :param train_file: Optional separate path for training data. If provided, overrides train_val_test_split.
        :param val_file: Optional separate path for validation data. Required if train_file is provided.
        :param test_file: Optional separate path for test data. Required if train_file is provided.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
        # Store normalization parameters for inference
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        
        # K-Fold specific attributes
        self.trainval_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.kfold: Optional[KFold] = None
        self.fold_indices: Optional[list] = None
        
        # Feature selection attributes
        self._selected_indices: Optional[np.ndarray] = None
        self._feature_scores: Optional[np.ndarray] = None
        self._feature_stages: Optional[list] = None  # For pipeline selection

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of unique classes in the dataset.
        """
        # This will be set after loading the data
        if hasattr(self, '_num_classes'):
            return self._num_classes
        return 2  # Default to binary classification

    @property
    def num_features(self) -> int:
        """Get the number of features (SNPs).

        :return: The number of SNP features in the dataset.
        """
        if hasattr(self, '_num_features'):
            return self._num_features
        return 0

    def _resolve_data_path(self, file_path: Optional[str]) -> Optional[str]:
        """Resolve a data file path relative to data_dir if data_dir is set.
        
        :param file_path: The file path to resolve. Can be absolute or relative.
        :return: The resolved absolute path, or None if file_path is None.
        """
        import os
        
        if file_path is None:
            return None
        
        # If path is already absolute, return as is
        if os.path.isabs(file_path):
            return file_path
        
        # If data_dir is specified AND we're NOT using split files, resolve path relative to it
        # (When using split files from train_file, ignore data_dir to avoid path doubling)
        if self.hparams.data_dir is not None and self.hparams.train_file is None:
            return os.path.join(self.hparams.data_dir, file_path)
        
        # Otherwise return path as is (relative to working directory)
        return file_path

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        import os
        
        # If using separate split files, check all of them
        if self.hparams.train_file is not None:
            train_file = self._resolve_data_path(self.hparams.train_file)
            if not os.path.exists(train_file):
                raise FileNotFoundError(
                    f"Training data file not found: {train_file}\n"
                    f"Please ensure the SNP data CSV file is available at the specified path."
                )
            val_file = self._resolve_data_path(self.hparams.val_file)
            if not os.path.exists(val_file):
                raise FileNotFoundError(
                    f"Validation data file not found: {val_file}\n"
                    f"Please ensure the SNP data CSV file is available at the specified path."
                )
            test_file = self._resolve_data_path(self.hparams.test_file)
            if not os.path.exists(test_file):
                raise FileNotFoundError(
                    f"Test data file not found: {test_file}\n"
                    f"Please ensure the SNP data CSV file is available at the specified path."
                )
        else:
            # Check single data file exists
            data_file = self._resolve_data_path(self.hparams.data_file)
            if not os.path.exists(data_file):
                raise FileNotFoundError(
                    f"Data file not found: {data_file}\n"
                    f"Please ensure the SNP data CSV file is available at the specified path."
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        import logging
        log = logging.getLogger(__name__)
        
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Check if separate split files are provided
            if self.hparams.train_file is not None and self.hparams.val_file is not None and self.hparams.test_file is not None:
                print("[SNP DataModule] Using separate split files...")
                log.info("Using separate split files for train/val/test")
                self._setup_separate_splits()
            else:
                print("[SNP DataModule] Loading full dataset...")
                log.info("Loading full dataset")
                # Load data
                data, labels = self._load_data()
                
                # Store number of features and classes
                self._num_features = data.shape[1]
                self._num_classes = len(torch.unique(labels))
                
                # Normalize if requested
                if self.hparams.normalize:
                    print("[SNP DataModule] Normalizing data (z-score normalization)...")
                    log.info("Normalizing data (z-score normalization)...")
                    data, self.mean, self.std = self._normalize_data(data)
                    print("[SNP DataModule] Data normalization complete")
                    log.info("Data normalization complete.")
                
                # Apply feature selection if requested
                if self.hparams.feature_selection is not None:
                    print(f"[SNP DataModule] Applying feature selection: {self.hparams.feature_selection.get('method', 'unknown')}...")
                    log.info(f"Applying feature selection: {self.hparams.feature_selection}")
                    data, self._selected_indices, self._feature_scores = self._select_features(data, labels)
                    print(f"[SNP DataModule] Feature selection complete. Selected {len(self._selected_indices)} features from {data.shape[1]}")
                    log.info(f"Feature selection complete. Selected {len(self._selected_indices)} features.")
                
                # Create full dataset
                print("[SNP DataModule] Creating dataset...")
                log.info("Creating dataset...")
                full_dataset = Dataset(data, labels)
                
                # Check if k-fold cross validation is requested
                if self.hparams.num_folds is not None and self.hparams.num_folds > 1:
                    print(f"[SNP DataModule] Setting up {self.hparams.num_folds}-fold cross validation (fold {self.hparams.current_fold})...")
                    log.info(f"Setting up {self.hparams.num_folds}-fold cross validation (fold {self.hparams.current_fold})...")
                    self._setup_kfold(full_dataset)
                    print(f"[SNP DataModule] K-fold setup complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
                    log.info(f"K-fold setup complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
                else:
                    print("[SNP DataModule] Setting up regular train/val/test split...")
                    log.info("Setting up regular train/val/test split...")
                    self._setup_regular_split(full_dataset)
                    print(f"[SNP DataModule] Split complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
                    log.info(f"Split complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
    
    
    def _setup_separate_splits(self) -> None:
        """Setup using separate pre-split data files for train/val/test.
        
        Loads and processes train, validation, and test data from separate CSV files.
        """
        import logging
        log = logging.getLogger(__name__)
        
        print("[SNP DataModule] Loading separate split files...")
        log.info("Loading separate split files for train/val/test")
        
        # Load training data
        print("[SNP DataModule] Loading training data...")
        train_data, train_labels = self._load_data_from_file(
            self._resolve_data_path(self.hparams.train_file),
            split_name="train"
        )
        
        # Load validation data
        print("[SNP DataModule] Loading validation data...")
        val_data, val_labels = self._load_data_from_file(
            self._resolve_data_path(self.hparams.val_file),
            split_name="validation"
        )
        
        # Load test data
        print("[SNP DataModule] Loading test data...")
        test_data, test_labels = self._load_data_from_file(
            self._resolve_data_path(self.hparams.test_file),
            split_name="test"
        )
        
        # Combine all data for computing statistics
        # Data tensors have shape (samples, features), so concatenate along dim=0 (samples)
        all_data = torch.cat([train_data, val_data, test_data], dim=0)
        all_labels = torch.cat([train_labels, val_labels, test_labels], dim=0)
        
        # Store number of features and classes
        self._num_features = all_data.shape[1]
        self._num_classes = len(torch.unique(all_labels))
        
        # Normalize if requested (compute statistics from all data)
        if self.hparams.normalize:
            print("[SNP DataModule] Normalizing data (z-score normalization)...")
            log.info("Normalizing data (z-score normalization)...")
            train_data, self.mean, self.std = self._normalize_data(train_data)
            val_data = (val_data - self.mean) / (self.std + 1e-8)
            test_data = (test_data - self.mean) / (self.std + 1e-8)
            print("[SNP DataModule] Data normalization complete")
            log.info("Data normalization complete.")
        
        # Apply feature selection if requested (learn from training data only)
        if self.hparams.feature_selection is not None:
            print(f"[SNP DataModule] Applying feature selection: {self.hparams.feature_selection.get('method', 'unknown')}...")
            log.info(f"Applying feature selection: {self.hparams.feature_selection}")
            train_data, self._selected_indices, self._feature_scores = self._select_features(train_data, train_labels)
            # Apply same feature selection to val and test data
            if self._selected_indices is not None:
                val_data = val_data[:, self._selected_indices]
                test_data = test_data[:, self._selected_indices]
            print(f"[SNP DataModule] Feature selection complete. Selected {len(self._selected_indices)} features")
            log.info(f"Feature selection complete. Selected {len(self._selected_indices)} features.")
        
        # Create datasets from tensors
        self.data_train = Dataset(train_data, train_labels)
        self.data_val = Dataset(val_data, val_labels)
        self.data_test = Dataset(test_data, test_labels)
        
        print(f"[SNP DataModule] Split setup complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
        log.info(f"Split setup complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
    
    def _load_data_from_file(self, file_path: str, split_name: str = "data") -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single data file and return as tensors.
        
        :param file_path: Path to the CSV file
        :param split_name: Name of the split for logging
        :return: Tuple of (data_tensor, labels_tensor)
        """
        import logging
        log = logging.getLogger(__name__)
        
        # Read CSV file with memory-efficient method
        header = 0 if self.hparams.has_header else None
        index_col = 0 if self.hparams.has_index else None
        
        try:
            # Try pandas with chunked reading for large files (more memory efficient)
            chunks = []
            for chunk in pd.read_csv(
                file_path, 
                header=header, 
                index_col=index_col,
                low_memory=True,
                chunksize=10000,  # Read in 10k row chunks
                # Don't specify dtype globally - let pandas auto-detect, will convert later
            ):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=False)
        except (pd.errors.ParserError, MemoryError, ValueError):
            # Fallback: use numpy for ultra-memory-efficient loading
            print(f"[SNP DataModule] Pandas failed, using numpy for memory-efficient loading...")
            data_array = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=np.float32)
            # This approach loses column/index names but handles memory better
            df = pd.DataFrame(data_array)
        
        print(f"[SNP DataModule] {split_name.capitalize()} CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        log.info(f"{split_name.capitalize()} CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Load or extract labels
        if self.hparams.label_file is not None:
            # Labels in separate file
            label_file = self._resolve_data_path(self.hparams.label_file)
            labels_df = pd.read_csv(label_file)
            labels = labels_df.values.flatten()
        elif self.hparams.labels_in_last_row:
            # Labels are in the last row of the CSV file
            labels = df.iloc[-1].values
            df = df.iloc[:-1]
        elif self.hparams.label_row_name is not None:
            # Labels in data file as a specific named row
            labels = df.loc[self.hparams.label_row_name].values
            df = df.drop(self.hparams.label_row_name)
        else:
            # Try to extract labels from the last column (per-sample labels)
            # This is the format for split files: data columns + label column
            last_col_name = df.columns[-1]
            labels = df[last_col_name].values
            df = df.drop(columns=[last_col_name])
        
        # Convert string labels to numeric
        labels = self._convert_labels_to_numeric(labels)
        
        # Remove the label column from the data
        # Now df is (samples, features) where features are SNPs
        
        # Convert to numeric (keep as samples × features)
        data_values = df.values.astype(float)
        data_values = np.nan_to_num(data_values, nan=0.0)
        
        # Convert to tensors - keep as (samples, features)
        data_tensor = torch.from_numpy(data_values).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        return data_tensor, labels_tensor
    
    def _setup_regular_split(self, full_dataset: Dataset) -> None:
        """Setup regular train/val/test split.
        
        :param full_dataset: The full dataset to split.
        """
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.hparams.train_val_test_split[0] * total_size)
        val_size = int(self.hparams.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
    
    def _setup_kfold(self, full_dataset: Dataset) -> None:
        """Setup k-fold cross validation split.
        
        :param full_dataset: The full dataset to split.
        """
        total_size = len(full_dataset)
        
        # Calculate test set size based on train_val_test_split[2]
        test_size = int(self.hparams.train_val_test_split[2] * total_size)
        trainval_size = total_size - test_size
        
        # Generate random indices for splitting
        torch.manual_seed(42)
        indices = torch.randperm(total_size).tolist()
        
        trainval_indices = indices[:trainval_size]
        test_indices = indices[trainval_size:]
        
        # Create train+val and test datasets
        self.trainval_dataset = Subset(full_dataset, trainval_indices)
        self.test_dataset = Subset(full_dataset, test_indices)
        
        # Initialize k-fold splitter
        self.kfold = KFold(n_splits=self.hparams.num_folds, shuffle=True, random_state=42)
        
        # Generate all fold indices
        self.fold_indices = list(self.kfold.split(trainval_indices))
        
        # Setup current fold
        self._apply_fold(self.hparams.current_fold)
    
    def _apply_fold(self, fold_idx: int) -> None:
        """Apply a specific fold to create train/val split.
        
        :param fold_idx: Index of the fold to apply (0 to num_folds-1).
        """
        if self.fold_indices is None:
            raise RuntimeError("K-Fold indices not initialized. Call setup() first.")
        
        if fold_idx >= self.hparams.num_folds:
            raise ValueError(f"Fold index {fold_idx} is out of range (0 to {self.hparams.num_folds - 1})")
        
        # Get train and val indices for this fold
        train_idx, val_idx = self.fold_indices[fold_idx]
        
        # Create subsets for train and val
        self.data_train = Subset(self.trainval_dataset, train_idx.tolist())
        self.data_val = Subset(self.trainval_dataset, val_idx.tolist())
        self.data_test = self.test_dataset

    def _select_features(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Apply feature selection to the data.
        
        :param data: Input data tensor of shape (n_samples, n_features).
        :param labels: Label tensor of shape (n_samples,).
        :return: Tuple of (selected_data, selected_indices, scores).
        """
        from src.data.feature_selection import select_features
        
        if self.hparams.feature_selection is None:
            raise RuntimeError("feature_selection not configured.")
        
        # Extract configuration and convert from OmegaConf if needed
        from omegaconf import OmegaConf
        config = self.hparams.feature_selection
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        
        method = config.get('method')
        if method is None:
            raise ValueError("feature_selection config must include 'method' key.")
        
        # For pipeline method, handle stages
        if method.lower() == 'pipeline':
            stages = config.get('stages')
            if stages is None:
                raise ValueError("Pipeline method requires 'stages' key in feature_selection config.")
            
            # Create container for stage info
            stage_info = []
            method_params = {'stages': stages, '_stage_info_out': stage_info}
            
            # Apply pipeline selection
            selected_data, selected_indices, scores = select_features(
                data, labels, method, **method_params
            )
            
            # Store stage information
            self._feature_stages = stage_info
            
        else:
            # Single-stage selection (existing behavior)
            # Extract k and other params
            k = config.get('k', None)
            
            # Get method-specific parameters
            method_params = {key: val for key, val in config.items() 
                            if key not in ['method', 'k']}
            
            # Apply selection
            selected_data, selected_indices, scores = select_features(
                data, labels, method, k=k, **method_params
            )
            
            self._feature_stages = None  # Not a pipeline
        
        return selected_data, selected_indices, scores

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load SNP data from CSV file.

        :return: Tuple of (data_tensor, labels_tensor).
        """
        import logging
        log = logging.getLogger(__name__)
        
        # Resolve data file path
        data_file = self._resolve_data_path(self.hparams.data_file)
        
        print(f"[SNP DataModule] Loading SNP data from {data_file}...")
        log.info(f"Loading SNP data from {data_file}...")
        
        # Determine header and index parameters
        header = 0 if self.hparams.has_header else None
        index_col = 0 if self.hparams.has_index else None
        
        # Load data with low_memory=False to handle mixed types
        # This is expected for SNP data with labels in the last row
        print("[SNP DataModule] Reading CSV file (this may take a while for large datasets)...")
        log.info("Reading CSV file (this may take a while for large datasets)...")
        df = pd.read_csv(
            data_file, 
            header=header, 
            index_col=index_col,
            low_memory=False
        )
        print(f"[SNP DataModule] CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        log.info(f"CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Load or extract labels
        if self.hparams.label_file is not None:
            # Labels in separate file - resolve label_file path
            label_file = self._resolve_data_path(self.hparams.label_file)
            labels_df = pd.read_csv(label_file)
            labels = labels_df.values.flatten()
        elif self.hparams.labels_in_last_row:
            # Labels are in the last row of the CSV file (case/control for each sample)
            labels = df.iloc[-1].values  # Get last row (samples as columns)
            df = df.iloc[:-1]  # Remove last row from data
        elif self.hparams.label_row_name is not None:
            # Labels in data file as a specific named row
            labels = df.loc[self.hparams.label_row_name].values
            df = df.drop(self.hparams.label_row_name)
        else:
            raise ValueError(
                "Either 'label_file', 'labels_in_last_row', or 'label_row_name' must be provided to identify labels."
            )
        
        # Convert string labels to numeric (case=1, control=0)
        print("[SNP DataModule] Converting labels to numeric...")
        log.info("Converting labels to numeric...")
        labels = self._convert_labels_to_numeric(labels)
        
        # Transpose: samples should be rows, features should be columns
        # Original: SNPs (features) as rows, samples as columns
        # After transpose: samples as rows, SNPs (features) as columns
        print("[SNP DataModule] Transposing data (samples as rows, SNPs as columns)...")
        log.info("Transposing data (samples as rows, SNPs as columns)...")
        data = df.T
        
        # Convert data to numeric, coercing any non-numeric values to NaN
        # Then replace NaN with 0 (or could use mean imputation)
        print(f"[SNP DataModule] Converting to numeric ({data.shape[0]} samples × {data.shape[1]} features)...")
        log.info(f"Converting to numeric and handling missing values ({data.shape[0]} samples × {data.shape[1]} features)...")
        # More efficient: convert directly without creating intermediate DataFrame
        data_values = data.values.astype(float)
        # Replace any NaN with 0
        data_values = np.nan_to_num(data_values, nan=0.0)
        
        # Convert to tensors
        print("[SNP DataModule] Converting to PyTorch tensors...")
        log.info("Converting to PyTorch tensors...")
        data_tensor = torch.tensor(data_values, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        print(f"[SNP DataModule] Data loaded: {data_tensor.shape[0]} samples, {data_tensor.shape[1]} features, {len(torch.unique(labels_tensor))} classes")
        log.info(f"Data loading complete: {data_tensor.shape[0]} samples, {data_tensor.shape[1]} features, {len(torch.unique(labels_tensor))} classes")
        
        return data_tensor, labels_tensor

    def _convert_labels_to_numeric(self, labels: np.ndarray) -> np.ndarray:
        """Convert string labels to numeric values.

        Converts 'case' to 1 and 'control' to 0.
        If labels are already numeric, returns them unchanged.

        :param labels: Array of labels (can be strings or numbers).
        :return: Array of numeric labels.
        """
        # Convert to numpy array if it's a pandas series or has StringDtype
        if hasattr(labels, 'to_numpy'):
            labels = labels.to_numpy()
        
        # If labels are already numeric, return as is
        try:
            if np.issubdtype(labels.dtype, np.number):
                return labels
        except (TypeError, AttributeError):
            # Handle pandas StringDtype and other non-standard dtypes
            pass
        
        # Convert to lowercase strings for case-insensitive matching
        labels_lower = np.array([str(label).lower().strip() for label in labels])
        
        # Create numeric labels array
        numeric_labels = np.zeros(len(labels), dtype=int)
        
        # Map case to 1, control to 0
        for i, label in enumerate(labels_lower):
            if label == 'case':
                numeric_labels[i] = 1
            elif label == 'control':
                numeric_labels[i] = 0
            else:
                # Try to convert to int if it's a number string
                try:
                    numeric_labels[i] = int(label)
                except ValueError:
                    raise ValueError(
                        f"Unknown label '{labels[i]}' at index {i}. "
                        f"Expected 'case', 'control', or numeric values."
                    )
        
        return numeric_labels

    def _normalize_data(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize data using z-score normalization.

        :param data: Input data tensor.
        :return: Tuple of (normalized_data, mean, std).
        """
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        
        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        normalized_data = (data - mean) / std
        
        return normalized_data, mean.squeeze(), std.squeeze()

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        state = {}
        if self.mean is not None:
            state['mean'] = self.mean
        if self.std is not None:
            state['std'] = self.std
        if hasattr(self, '_num_features'):
            state['num_features'] = self._num_features
        if hasattr(self, '_num_classes'):
            state['num_classes'] = self._num_classes
        if self.hparams.num_folds is not None:
            state['current_fold'] = self.hparams.current_fold
        # Save feature selection state
        if self._selected_indices is not None:
            state['selected_indices'] = self._selected_indices
        if self._feature_scores is not None:
            state['feature_scores'] = self._feature_scores
        if self._feature_stages is not None:
            state['feature_stages'] = self._feature_stages
        if self.hparams.feature_selection is not None:
            state['feature_selection_config'] = self.hparams.feature_selection
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        if 'mean' in state_dict:
            self.mean = state_dict['mean']
        if 'std' in state_dict:
            self.std = state_dict['std']
        if 'num_features' in state_dict:
            self._num_features = state_dict['num_features']
        if 'num_classes' in state_dict:
            self._num_classes = state_dict['num_classes']
        if 'current_fold' in state_dict:
            self.hparams.current_fold = state_dict['current_fold']
        # Load feature selection state
        if 'selected_indices' in state_dict:
            self._selected_indices = state_dict['selected_indices']
        if 'feature_scores' in state_dict:
            self._feature_scores = state_dict['feature_scores']
        if 'feature_stages' in state_dict:
            self._feature_stages = state_dict['feature_stages']
        if 'feature_selection_config' in state_dict:
            self.hparams.feature_selection = state_dict['feature_selection_config']


if __name__ == "__main__":
    _ = DataModule()
