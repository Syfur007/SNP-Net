"""Stacked LSTM model for SNP classification.

This module implements a Stacked LSTM neural network with configurable
hidden dimensions for each layer, allowing for more expressive hierarchical
sequence modeling compared to standard LSTM.
"""

import torch
from torch import nn


class StackedLSTMNet(nn.Module):
    """A Stacked LSTM neural network for SNP data classification.

    This model extends the basic LSTM approach by allowing different hidden
    dimensions for each LSTM layer, creating a hierarchical representation.
    Each layer can learn features at different levels of abstraction.
    """

    def __init__(
        self,
        window_size: int = 50,
        hidden_sizes: list[int] = None,
        output_size: int = 2,
        dropout: float = 0.3,
        input_size: int = None,  # Optional, for backward compatibility
    ) -> None:
        """Initialize a `StackedLSTMNet` module.

        :param window_size: Size of each window/chunk for sequence creation.
        :param hidden_sizes: List of hidden dimensions for each LSTM layer.
            Default: [128, 64, 32] for 3 layers with decreasing dimensions.
        :param output_size: The number of output features (classes).
        :param dropout: Dropout probability between LSTM layers and in classifier.
        :param input_size: Optional. The total number of SNP features. If None, will be inferred from data.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self._input_size = input_size
        self.window_size = window_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout
        self.num_layers = len(hidden_sizes)

        # These will be calculated dynamically if input_size is not provided
        if input_size is not None:
            self.seq_len = (input_size + window_size - 1) // window_size
            self.padded_size = self.seq_len * window_size
        else:
            self.seq_len = None
            self.padded_size = None

        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_dim = window_size

        for i, hidden_size in enumerate(hidden_sizes):
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=1,  # Single layer per LSTM module
                batch_first=True,
                dropout=0,  # Handle dropout separately between layers
            )
            self.lstm_layers.append(lstm)
            input_dim = hidden_size

            # Add dropout between layers (except after the last layer)
            if i < len(hidden_sizes) - 1:
                self.lstm_layers.append(nn.Dropout(dropout))

        # Classifier from final hidden state
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, output_size),
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

        # Pass through stacked LSTM layers
        h_n = None
        for layer in self.lstm_layers:
            if isinstance(layer, nn.LSTM):
                # LSTM layer
                # x shape: (batch, seq_len, input_dim)
                # lstm_out shape: (batch, seq_len, hidden_size)
                # h_n shape: (1, batch, hidden_size)
                x, (h_n, c_n) = layer(x)
            elif isinstance(layer, nn.Dropout):
                # Dropout layer - apply to sequence output
                x = layer(x)

        # Use the final hidden state from the last LSTM layer
        # h_n[-1] shape: (batch, hidden_size)
        if h_n is not None:
            final_hidden = h_n[-1]
        else:
            # Fallback: use last time step if h_n is not available
            final_hidden = x[:, -1, :]

        # Pass through classifier
        out = self.fc(final_hidden)

        return out


if __name__ == "__main__":
    # Test the model
    print("Testing StackedLSTMNet...")

    # Test with dynamic input sizing
    model = StackedLSTMNet(
        window_size=50, hidden_sizes=[128, 64, 32], output_size=2, dropout=0.3
    )

    # Simulate SNP data
    batch_size = 32
    num_snps = 1000
    x = torch.randn(batch_size, num_snps)

    print(f"Input shape: {x.shape}")
    print(f"Hidden sizes: {model.hidden_sizes}")
    print(f"Number of LSTM layers: {model.num_layers}")

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
    model2 = StackedLSTMNet(
        window_size=100,
        hidden_sizes=[256, 128, 64, 32],
        output_size=3,
        input_size=num_snps,
    )
    logits3 = model2(x)
    print(f"Pre-built model (4 layers) output shape: {logits3.shape}")
    assert logits3.shape == (batch_size, 3), "Incorrect output shape for pre-built model"

    # Test with single layer (should also work)
    model3 = StackedLSTMNet(window_size=50, hidden_sizes=[128], output_size=2)
    logits4 = model3(x)
    print(f"Single layer model output shape: {logits4.shape}")
    assert logits4.shape == (batch_size, 2), "Incorrect output shape for single layer"

    # Test parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nStacked LSTM model parameters: {params:,}")

    # Compare layers
    print(f"\nModel architecture:")
    for i, layer in enumerate(model.lstm_layers):
        if isinstance(layer, nn.LSTM):
            print(
                f"  Layer {i}: LSTM (input={layer.input_size}, hidden={layer.hidden_size})"
            )
        else:
            print(f"  Layer {i}: Dropout(p={layer.p})")

    print("\n✓ All tests passed!")
