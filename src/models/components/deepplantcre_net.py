"""
DeepPlantCRE: Transformer-CNN Hybrid for Gene Expression Modeling
==================================================================

Transformer-CNN hybrid architecture specifically designed for plant gene expression
modeling and cis-regulatory element prediction. Adapted for genomic classification tasks.

Reference:
- DeepPlantCRE: Transformer-CNN Hybrid (arXiv:2505.09883v1, 2024)
- Wu et al. "DeepPlantCRE: A Transformer-CNN Hybrid Framework for Plant Gene Expression 
  Modeling and Cross-Species Generalization"

Architecture:
- Single-layer Transformer encoder with 1 attention head (d_model=4)
- 3 cascaded Residual CNN blocks (4→64→128→32 channels)
- 3 Dense layers for classification
- BCEWithLogitsLoss for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""
    def __init__(self, d_model: int, dropout: float = 0.25, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualCNNBlock(nn.Module):
    """
    Residual CNN block following DeepPlantCRE architecture.
    
    Structure:
    - Conv1D + ReLU → Conv1D + ReLU (main path)
    - Conv1D (1x1 residual connection)
    - BatchNorm + Add + ReLU
    - MaxPool1D + Dropout
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8, dropout: float = 0.25):
        super().__init__()
        
        # Main path: two 8-kernel convolutions
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Batch normalization for the second convolution
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Residual connection: 1x1 conv for channel adjustment if needed
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_channels, length)
        Returns:
            Tensor of shape (batch, out_channels, length//2)
        """
        # Main path: Conv → ReLU → Conv → ReLU
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        
        # Apply batch norm
        out = self.bn(out)
        
        # Residual connection with channel adjustment
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        else:
            residual = x
        
        # Handle pooling for residual connection
        residual = self.pool(residual)
        
        # Match dimensions if needed (for odd output sizes)
        if out.size(2) != residual.size(2):
            min_size = min(out.size(2), residual.size(2))
            out = out[:, :, :min_size]
            residual = residual[:, :, :min_size]
        
        # Add residual and apply pooling to main path
        out = self.pool(out + residual)
        out = F.relu(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out


class DeepPlantCRENet(nn.Module):
    """
    DeepPlantCRE: Transformer-CNN Hybrid Architecture
    
    Processes DNA sequences (or sequence-like genomic data) through:
    1. Transformer encoder (1 layer, 1 head, d_model=4)
    2. Three cascaded residual CNN blocks (4→64→128→32)
    3. Fully connected classification head
    
    Args:
        input_size (int): Total input features. If None, dynamically inferred.
        num_snps (int): Number of SNPs/positions in sequence (default: 1500 for 1.5kbp DNA seq)
        encoding_dim (int): Encoding dimension per SNP (default: 4 for ATCG one-hot)
        d_model (int): Transformer embedding dimension (default: 4, fixed for DeepPlantCRE)
        nhead (int): Number of attention heads (default: 1, fixed for DeepPlantCRE)
        num_transformer_layers (int): Number of transformer layers (default: 1, fixed for DeepPlantCRE)
        d_ff (int): Feed-forward dimension in transformer (default: 2048)
        cnn_channels (list): Channel dimensions for 3 CNN blocks (default: [64, 128, 32])
        dense_dims (list): Hidden dimensions for dense layers
        dropout (float): Dropout rate throughout (default: 0.25)
        output_size (int): Number of output classes (default: 2 for binary classification with CrossEntropyLoss)
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        input_size: int = None,
        num_snps: int = 1500,
        encoding_dim: int = 4,
        d_model: int = 4,
        nhead: int = 1,
        num_transformer_layers: int = 1,
        d_ff: int = 2048,
        cnn_channels: list = None,
        dense_dims: list = None,
        dropout: float = 0.25,
        output_size: int = 2,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize DeepPlantCRENet module."""
        super().__init__()
        
        # Use default channel configuration if not provided
        if cnn_channels is None:
            cnn_channels = [64, 128, 32]
        
        if dense_dims is None:
            dense_dims = [256, 128]
        
        self._input_size = input_size
        self.num_snps = num_snps
        self.encoding_dim = encoding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.d_ff = d_ff
        self.cnn_channels = cnn_channels
        self.dense_dims = dense_dims
        self.dropout_rate = dropout
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # ============ Transformer Encoder ============
        # Input projection: encoding_dim → d_model
        self.embedding = nn.Linear(encoding_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder: 1 layer, 1 head, d_ff=2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False  # Apply LayerNorm after residual connection
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # ============ Residual CNN Blocks ============
        # 3 cascaded blocks: 4→64, 64→128, 128→32
        self.cnn_blocks = nn.ModuleList()
        in_channels = d_model
        for out_channels in cnn_channels:
            self.cnn_blocks.append(
                ResidualCNNBlock(in_channels, out_channels, kernel_size=8, dropout=dropout)
            )
            in_channels = out_channels
        
        # ============ Classification Head ============
        # Flatten output from CNN
        # We'll compute the flattened size dynamically based on pooling operations
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers: cnn_channels[-1] → dense_dims[0] → dense_dims[1] → output_size
        dense_layers = []
        prev_dim = cnn_channels[-1]
        
        for hidden_dim in dense_dims:
            dense_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for BCEWithLogitsLoss)
        dense_layers.append(nn.Linear(prev_dim, output_size))
        
        self.classifier = nn.Sequential(*dense_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepPlantCRE architecture.
        
        Args:
            x: Input tensor of shape (batch, num_features)
               - If num_features matches num_snps * encoding_dim, will be reshaped to sequence
               - Otherwise, will be truncated or padded
        
        Returns:
            Output tensor of shape (batch, output_size)
        """
        batch_size = x.size(0)
        total_features = x.size(1)
        
        # Calculate how many SNPs we can fit
        actual_num_snps = min(self.num_snps, total_features // self.encoding_dim)
        
        if actual_num_snps == 0:
            # If we can't fit any SNPs, use all features as individual SNPs
            actual_num_snps = total_features
            effective_encoding_dim = 1
        else:
            effective_encoding_dim = self.encoding_dim
        
        # Truncate to fit available features
        features_to_use = actual_num_snps * effective_encoding_dim
        x = x[:, :features_to_use]
        
        # Reshape to sequence: (batch, num_snps, encoding_dim)
        x = x.view(batch_size, actual_num_snps, effective_encoding_dim)
        
        # ============ Transformer Encoder ============
        # Input embedding: (batch, num_snps, encoding_dim) → (batch, num_snps, d_model)
        x = self.embedding(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding with gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
        else:
            x = self.transformer_encoder(x)  # (batch, num_snps, d_model)
        
        # Transpose for CNN: (batch, d_model, num_snps)
        x = x.transpose(1, 2)
        
        # ============ CNN Blocks ============
        for cnn_block in self.cnn_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(cnn_block, x, use_reentrant=False)
            else:
                x = cnn_block(x)
        
        # ============ Classification Head ============
        # Global average pooling: (batch, cnn_channels[-1], reduced_seq_len) → (batch, cnn_channels[-1])
        x = self.global_pool(x).squeeze(-1)
        
        # Classification: (batch, cnn_channels[-1]) → (batch, output_size)
        output = self.classifier(x)
        
        return output


if __name__ == "__main__":
    # Test the model with DNA sequence-like input
    print("=" * 60)
    print("DeepPlantCRE Architecture Test")
    print("=" * 60)
    
    # Create model with default parameters (1500 bp, 4D one-hot encoding)
    model = DeepPlantCRENet(
        num_snps=1500,
        encoding_dim=4,
        d_model=4,
        nhead=1,
        num_transformer_layers=1,
        cnn_channels=[64, 128, 32],
        dropout=0.25,
        output_size=2  # Binary classification with CrossEntropyLoss
    )
    
    # Test with random input (batch_size=32, 1500*4=6000 features)
    batch_size = 32
    num_features = 1500 * 4  # 1500 positions × 4 one-hot
    x = torch.randn(batch_size, num_features)
    
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with SNP data (flat features)
    print("\n" + "=" * 60)
    print("Test with SNP-like data (different feature size)")
    print("=" * 60)
    
    x_snp = torch.randn(batch_size, 10000)  # 10,000 features
    output_snp = model(x_snp)
    
    print(f"\nInput shape: {x_snp.shape}")
    print(f"Output shape: {output_snp.shape}")
    
    print("\n" + "=" * 60)
    print("Test passed! ✓")
    print("=" * 60)
