from pathlib import Path

import pytest
import torch

from src.data.datamodule import DataModule


@pytest.mark.parametrize("batch_size", [32, 64])
def test_snp_datamodule(batch_size: int) -> None:
    """Tests `DataModule` to verify that it can load data correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_file = "data/snp_data.csv"

    # Test without k-fold
    dm = DataModule(data_file=data_file, batch_size=batch_size, num_folds=None)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_file).exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints > 0  # Should have data loaded

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) <= batch_size  # May be less on last batch
    assert len(y) <= batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("num_folds", [3, 5])
def test_snp_datamodule_kfold(num_folds: int) -> None:
    """Tests `DataModule` with k-fold cross validation.

    :param num_folds: Number of folds for cross validation.
    """
    data_file = "data/snp_data.csv"
    batch_size = 32

    # Test with k-fold
    dm = DataModule(
        data_file=data_file, 
        batch_size=batch_size, 
        num_folds=num_folds,
        current_fold=0
    )
    dm.prepare_data()
    dm.setup()

    assert dm.data_train and dm.data_val and dm.data_test
    
    # Test set should be consistent
    test_size_fold0 = len(dm.data_test)
    
    # Check different fold
    dm2 = DataModule(
        data_file=data_file,
        batch_size=batch_size,
        num_folds=num_folds,
        current_fold=1
    )
    dm2.setup()
    
    # Test set should be same size across folds
    assert len(dm2.data_test) == test_size_fold0
    
    # Train/val sizes should be similar but data should be different
    assert abs(len(dm.data_train) - len(dm2.data_train)) <= 1  # Allow for rounding differences
