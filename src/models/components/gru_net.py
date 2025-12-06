"""GRU model for SNP classification.

This module implements a GRU (Gated Recurrent Unit) neural network that processes
SNP data as sequences by dividing the flat feature vector into windows.
"""

import torch
from torch import nn


class GRUNet(nn.Module):
    """A GRU-based neural network for SNP data classification.

    This model divides the flat SNP feature vector into chunks/windows
    to create a sequence for GRU processing. GRU is similar to LSTM but
    has fewer parameters and can be faster to train.
    """

    def __init__(
        self,
        window_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.3,
        input_size: int = None,  # Optional, for backward compatibility
    ) -> None:
        """Initialize a `GRUNet` module.

        :param window_size: Size of each window/chunk for sequence creation.
        :param hidden_size: The number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param output_size: The number of output features (classes).
        :param dropout: Dropout probability between GRU layers.
        :param input_size: Optional. The total number of SNP features. If None, will be inferred from data.
        """
        super().__init__()

        self._input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        # These will be calculated dynamically if input_size is not provided
        if input_size is not None:
            self.seq_len = (input_size + window_size - 1) // window_size
            self.padded_size = self.seq_len * window_size
        else:
            self.seq_len = None
            self.padded_size = None

        self.gru = nn.GRU(
            input_size=window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor of shape (batch, num_features).
        :return: A tensor of predictions of shape (batch, output_size).
        """
        batch_size = x.size(0)
        actual_input_size = x.size(1)

        # Recalculate sequence length based on actual input size
        seq_len = (actual_input_size + self.window_size - 1) // self.window_size
        padded_size = seq_len * self.window_size

        # Pad if necessary
        if actual_input_size < padded_size:
            padding = torch.zeros(
                batch_size, padded_size - actual_input_size, device=x.device
            )
            x = torch.cat([x, padding], dim=1)
        elif actual_input_size > padded_size:
            # Truncate if input is larger than expected
            x = x[:, :padded_size]

        # Reshape to sequence: (batch, seq_len, window_size)
        x = x.view(batch_size, seq_len, self.window_size)

        # GRU forward pass
        # gru_out shape: (batch, seq_len, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        gru_out, h_n = self.gru(x)

        # Use the last hidden state from the last layer
        # h_n[-1] shape: (batch, hidden_size)
        out = self.fc(h_n[-1])

        return out


if __name__ == "__main__":
    # Test the model
    print("Testing GRUNet...")

    # Test with dynamic input sizing
    model = GRUNet(window_size=50, hidden_size=128, num_layers=2, output_size=2)

    # Simulate SNP data
    batch_size = 32
    num_snps = 1000
    x = torch.randn(batch_size, num_snps)

    print(f"Input shape: {x.shape}")

    # Test forward pass
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 2), "Incorrect output shape"

    # Test with different input size
    x2 = torch.randn(16, 2500)
    logits2 = model(x2)
    print(f"Different input shape: {x2.shape} -> Output: {logits2.shape}")
    assert logits2.shape == (16, 2), "Incorrect output shape for different input"

    # Test with pre-specified input size
    model2 = GRUNet(
        window_size=100, hidden_size=64, num_layers=3, output_size=3, input_size=num_snps
    )
    logits3 = model2(x)
    print(f"Pre-built model output shape: {logits3.shape}")
    assert logits3.shape == (batch_size, 3), "Incorrect output shape for pre-built model"

    # Test parameter count comparison with LSTM
    gru_params = sum(p.numel() for p in model.parameters())
    print(f"\nGRU model parameters: {gru_params:,}")

    print("\n✓ All tests passed!")
