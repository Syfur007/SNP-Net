from typing import Tuple

import torch
from torch.utils.data import Dataset


class SNPDataset(Dataset):
    """Custom Dataset for SNP data stored in CSV format.
    
    The CSV file should have:
    - Samples as columns
    - Features (SNPs) as rows
    - Optional: First column as feature names/IDs
    - Optional: First row as sample names/IDs
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Initialize SNPDataset.

        :param data: Tensor of shape (n_samples, n_features).
        :param labels: Tensor of shape (n_samples,).
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample and its label.

        :param idx: Index of the sample.
        :return: Tuple of (sample, label).
        """
        return self.data[idx], self.labels[idx]
