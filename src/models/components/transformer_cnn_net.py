"""
Transformer-CNN Hybrid: DeepPlantCRE-inspired Architecture
===========================================================

Transformer-CNN hybrid architecture for genomic prediction, combining the power
of transformers for long-range dependencies with CNNs for local feature extraction.

Reference:
- DeepPlantCRE: Transformer-CNN hybrid (arXiv:2505.09883v1, 2024)
- State-of-the-art for plant regulatory element prediction, adapted for disease classification

Architecture:
- Transformer encoder for capturing long-range dependencies
- Hierarchical CNN for multi-scale local feature extraction
- Feature fusion for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualCNNBlock(nn.Module):
    """Residual CNN block with dropout and max pooling"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)
        
        # Shortcut connection: handles both channel and spatial dimension changes
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.MaxPool1d(2)
            )
        
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        
        # Shortcut path (includes pooling)
        residual = self.shortcut(x)
        
        # Match dimensions if needed (handle odd sizes)
        if out.size(2) != residual.size(2):
            # Crop the larger one to match the smaller
            min_size = min(out.size(2), residual.size(2))
            out = out[:, :, :min_size]
            residual = residual[:, :, :min_size]
        
        # Add residual and activate
        out = F.relu(out + residual)
        
        return out


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """Transformer encoder layer that stores attention weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attention_weights = None

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal: bool = False,
    ):
        x = src
        if self.norm_first:
            x = x + self._sa_block_with_weights(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_out = self._sa_block_with_weights(x, src_mask, src_key_padding_mask, is_causal)
            x = self.norm1(x + sa_out)
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block_with_weights(self, x, attn_mask, key_padding_mask, is_causal):
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attention_weights = attn_weights.detach()
        return self.dropout1(attn_output)


class TransformerCNNNet(nn.Module):
    """
    Transformer-CNN Hybrid for SNP-based genomic prediction
    
    This model combines transformers and CNNs by:
    1. Reshaping flat SNP vectors into sequences
    2. Applying transformer encoder for global context
    3. Using hierarchical CNN for multi-scale features
    4. Classifying based on pooled representations
    
    Args:
        input_size (int): Total number of SNP features (will be dynamically set)
        num_snps (int): Number of SNPs to use (default: 1000)
        encoding_dim (int): Encoding dimension per SNP (default: 1)
        d_model (int): Transformer embedding dimension
        nhead (int): Number of attention heads
        num_transformer_layers (int): Number of transformer layers
        cnn_channels (list): CNN channel dimensions
        dropout (float): Dropout rate
        output_size (int): Number of output classes
        use_gradient_checkpointing (bool): Use gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        input_size: int = None,
        num_snps: int = 1000,
        encoding_dim: int = 1,
        d_model: int = 64,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        cnn_channels: list = None,
        dropout: float = 0.25,
        output_size: int = 2,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize a `TransformerCNNNet` module.

        :param input_size: Optional. Total number of SNP features. If None, will be inferred from data.
        :param num_snps: Number of SNPs to use for sequence creation
        :param encoding_dim: Encoding dimension per SNP (1 for single value, 4 for ATCG, etc.)
        :param d_model: Transformer embedding dimension
        :param nhead: Number of attention heads
        :param num_transformer_layers: Number of transformer encoder layers
        :param cnn_channels: List of CNN channel dimensions (default: [64, 128, 32])
        :param dropout: Dropout rate
        :param output_size: Number of output classes
        :param use_gradient_checkpointing: Whether to use gradient checkpointing to reduce memory
        """
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [64, 128, 32]
        
        self._input_size = input_size
        self.num_snps = num_snps
        self.encoding_dim = encoding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.cnn_channels = cnn_channels
        self.dropout_rate = dropout
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.last_attention_weights = None
        
        # Input embedding
        self.embedding = nn.Linear(encoding_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Hierarchical CNN blocks with residual connections
        self.cnn_blocks = nn.ModuleList()
        in_channels = d_model
        for out_channels in cnn_channels:
            self.cnn_blocks.append(
                ResidualCNNBlock(in_channels, out_channels, kernel_size=8, dropout=dropout)
            )
            in_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
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
        
        # Input embedding: (batch, num_snps, encoding_dim) -> (batch, num_snps, d_model)
        x = self.embedding(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding (capture long-range dependencies)
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
            self.last_attention_weights = None
        else:
            x = self.transformer_encoder(x)  # (batch, num_snps, d_model)
            try:
                self.last_attention_weights = [
                    layer.last_attention_weights
                    for layer in self.transformer_encoder.layers
                    if hasattr(layer, "last_attention_weights")
                ]
            except Exception:
                self.last_attention_weights = None
        
        # Transpose for CNN: (batch, d_model, num_snps)
        x = x.transpose(1, 2)
        
        # Hierarchical CNN feature extraction
        for cnn_block in self.cnn_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(cnn_block, x, use_reentrant=False)
            else:
                x = cnn_block(x)
        
        # Global pooling: (batch, cnn_channels[-1], reduced_snps) -> (batch, cnn_channels[-1])
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        output = self.classifier(x)  # (batch, output_size)
        
        return output


if __name__ == "__main__":
    # Test the model
    model = TransformerCNNNet(
        num_snps=1000,
        encoding_dim=1,
        d_model=64,
        nhead=8,
        num_transformer_layers=2,
        cnn_channels=[64, 128, 32],
        dropout=0.25,
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
