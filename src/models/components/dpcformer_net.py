"""
DPCformer: Deep Pheno Correlation Former
=========================================

Hybrid CNN + Multi-Head Self-Attention architecture for SNP-based genomic prediction.

Reference: 
- DPCformer: CNN + Multi-Head Self-Attention (arXiv:2510.08662v1, 2024)
- State-of-the-art for crop genomics, adapted for disease classification

Architecture:
- Input projection for SNP encoding
- Residual CNN blocks for local feature extraction
- Multi-head self-attention for capturing long-range SNP interactions
- Feed-forward network with residual connections
- Global pooling and classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DPCformerNet(nn.Module):
    """
    Deep Pheno Correlation Former - Hybrid CNN + Transformer architecture
    
    This model is designed for SNP data where the input is a flat vector of SNP features.
    It handles the data by:
    1. Reshaping flat input into sequence format (num_snps, encoding_dim)
    2. Projecting to hidden dimension
    3. Applying CNN for local patterns
    4. Applying attention for long-range interactions
    5. Classifying the aggregated features
    
    Args:
        input_size (int): Total number of SNP features (will be dynamically set)
        num_snps (int): Number of SNPs to consider (default: 1000)
        encoding_dim (int): Encoding dimension per SNP (default: 1 for single value encoding)
        hidden_dim (int): Hidden dimension for attention mechanism
        num_heads (int): Number of attention heads
        num_cnn_layers (int): Number of CNN layers
        dropout (float): Dropout rate
        output_size (int): Number of output classes
        use_gradient_checkpointing (bool): Use gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        input_size: int = None,
        num_snps: int = 1000,
        encoding_dim: int = 1,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_cnn_layers: int = 3,
        dropout: float = 0.3,
        output_size: int = 2,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize a `DPCformerNet` module.

        :param input_size: Optional. Total number of SNP features. If None, will be inferred from data.
        :param num_snps: Number of SNPs to use for sequence creation
        :param encoding_dim: Encoding dimension per SNP (1 for single value, 3 for {0,1,2}, etc.)
        :param hidden_dim: Hidden dimension for attention mechanism
        :param num_heads: Number of attention heads
        :param num_cnn_layers: Number of CNN layers
        :param dropout: Dropout rate
        :param output_size: Number of output classes
        :param use_gradient_checkpointing: Whether to use gradient checkpointing to reduce memory
        """
        super().__init__()
        
        self._input_size = input_size
        self.num_snps = num_snps
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection: from encoding_dim to hidden_dim
        self.input_projection = nn.Linear(encoding_dim, hidden_dim)
        
        # Residual CNN blocks for local feature extraction
        self.cnn_blocks = nn.ModuleList([
            self._make_residual_cnn_block(hidden_dim, hidden_dim, kernel_size=3)
            for _ in range(num_cnn_layers)
        ])
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size)
        )
        
    def _make_residual_cnn_block(self, in_channels, out_channels, kernel_size=3):
        """Create a residual CNN block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels)
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
        
        # Input projection: (batch, num_snps, encoding_dim) -> (batch, num_snps, hidden_dim)
        x = self.input_projection(x)
        
        # CNN feature extraction
        # Transpose to (batch, hidden_dim, num_snps) for Conv1d
        x_cnn = x.transpose(1, 2)
        
        for cnn_block in self.cnn_blocks:
            residual = x_cnn
            if self.use_gradient_checkpointing and self.training:
                x_cnn = checkpoint(cnn_block, x_cnn, use_reentrant=False)
            else:
                x_cnn = cnn_block(x_cnn)
            x_cnn = F.relu(x_cnn + residual)  # Residual connection
        
        # Transpose back to (batch, num_snps, hidden_dim)
        x_cnn = x_cnn.transpose(1, 2)
        
        # Multi-head self-attention
        if self.use_gradient_checkpointing and self.training:
            attn_out = checkpoint(
                lambda q, k, v: self.attention(q, k, v)[0],
                x_cnn, x_cnn, x_cnn,
                use_reentrant=False
            )
        else:
            attn_out, _ = self.attention(x_cnn, x_cnn, x_cnn)
        x = self.norm1(x_cnn + attn_out)  # Residual + LayerNorm
        
        # Feed-forward network
        if self.use_gradient_checkpointing and self.training:
            ffn_out = checkpoint(self.ffn, x, use_reentrant=False)
        else:
            ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual + LayerNorm
        
        # Global pooling: (batch, num_snps, hidden_dim) -> (batch, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, num_snps)
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden_dim)
        
        # Classification
        output = self.classifier(x)  # (batch, output_size)
        
        return output


if __name__ == "__main__":
    # Test the model
    model = DPCformerNet(
        num_snps=1000,
        encoding_dim=1,
        hidden_dim=128,
        num_heads=8,
        num_cnn_layers=3,
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
