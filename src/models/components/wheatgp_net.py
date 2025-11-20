"""
WheatGP: CNN-LSTM Hybrid Architecture
======================================

CNN-LSTM architecture for genomic prediction, designed for capturing both local patterns
and long-range dependencies (epistatic effects) in SNP data.

Reference:
- WheatGP: CNN + LSTM for genomic prediction (Briefings in Bioinformatics, 2025)
- Best for capturing epistatic effects between distant SNPs

Architecture:
- Multi-layer CNN for local feature extraction (short-range dependencies)
- LSTM for modeling long-range dependencies and epistatic interactions
- Fully connected layers for final classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class WheatGPNet(nn.Module):
    """
    WheatGP CNN-LSTM architecture for SNP-based genomic prediction
    
    This model processes flat SNP vectors by:
    1. Reshaping into sequence format
    2. Applying CNN for local pattern extraction
    3. Using LSTM to capture long-range dependencies
    4. Classifying based on the learned representations
    
    Args:
        input_size (int): Total number of SNP features (will be dynamically set)
        num_snps (int): Number of SNPs to use (default: 1000)
        encoding_dim (int): Encoding dimension per SNP (default: 1 for single value)
        cnn_channels (list): List of channel sizes for CNN layers
        lstm_hidden (int): LSTM hidden dimension
        lstm_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        output_size (int): Number of output classes
        use_gradient_checkpointing (bool): Use gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        input_size: int = None,
        num_snps: int = 1000,
        encoding_dim: int = 1,
        cnn_channels: list = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 2,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize a `WheatGPNet` module.

        :param input_size: Optional. Total number of SNP features. If None, will be inferred from data.
        :param num_snps: Number of SNPs to use for sequence creation
        :param encoding_dim: Encoding dimension per SNP (1 for single value, 3 for {0,1,2}, etc.)
        :param cnn_channels: List of channel sizes for CNN layers (default: [64, 128, 256])
        :param lstm_hidden: LSTM hidden dimension
        :param lstm_layers: Number of LSTM layers
        :param dropout: Dropout rate
        :param output_size: Number of output classes
        :param use_gradient_checkpointing: Whether to use gradient checkpointing to reduce memory
        """
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        
        self._input_size = input_size
        self.num_snps = num_snps
        self.encoding_dim = encoding_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # CNN module for local feature extraction
        cnn_layers = []
        in_channels = encoding_dim
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        self.cnn_module = nn.Sequential(*cnn_layers)
        
        # LSTM module for capturing long-range dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a single forward pass through the network.

        :param x: The input tensor of shape (batch, num_features) - flat SNP vector
        :return: A tensor of predictions of shape (batch, output_size)
        """
        batch_size = x.size(0)
        total_features = x.size(1)
        
        # Calculate how many SNPs we can fit
        actual_num_snps = min(self.num_snps, total_features // self.encoding_dim)
        
        if actual_num_snps == 0:
            actual_num_snps = total_features  # Use all features as individual SNPs
            effective_encoding_dim = 1
        else:
            effective_encoding_dim = self.encoding_dim
        
        # Truncate or use available features
        features_to_use = actual_num_snps * effective_encoding_dim
        x = x[:, :features_to_use]
        
        # Reshape to (batch, num_snps, encoding_dim)
        x = x.view(batch_size, actual_num_snps, effective_encoding_dim)
        
        # CNN feature extraction
        # Transpose to (batch, encoding_dim, num_snps) for Conv1d
        x = x.transpose(1, 2)  # (batch, encoding_dim, num_snps)
        
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(self.cnn_module, x, use_reentrant=False)
        else:
            x = self.cnn_module(x)  # (batch, cnn_channels[-1], num_snps)
        
        # Transpose back to (batch, num_snps, cnn_channels[-1]) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM processing
        if self.use_gradient_checkpointing and self.training:
            lstm_out, (h_n, c_n) = checkpoint(
                self._lstm_forward,
                x,
                use_reentrant=False
            )
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from the final layer
        x = h_n[-1]  # (batch, lstm_hidden)
        
        # Classification
        output = self.classifier(x)  # (batch, output_size)
        
        return output
    
    def _lstm_forward(self, x):
        """Helper method for gradient checkpointing of LSTM."""
        return self.lstm(x)


if __name__ == "__main__":
    # Test the model
    model = WheatGPNet(
        num_snps=1000,
        encoding_dim=1,
        cnn_channels=[64, 128, 256],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
        output_size=2
    )
    
    # Test with random input
    batch_size = 32
    num_features = 10000
    x = torch.randn(batch_size, num_features)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
