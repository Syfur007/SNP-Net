import torch
from torch import nn


class LSTMNet(nn.Module):
    """An LSTM-based neural network for SNP data classification.
    
    This model divides the flat SNP feature vector into chunks/windows
    to create a sequence for LSTM processing.
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
        """Initialize a `LSTMNet` module.

        :param window_size: Size of each window/chunk for sequence creation.
        :param hidden_size: The number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param output_size: The number of output features (classes).
        :param dropout: Dropout probability between LSTM layers.
        :param input_size: Optional. The total number of SNP features. If None, will be inferred from data.
        """
        super().__init__()

        self._input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # These will be calculated dynamically if input_size is not provided
        if input_size is not None:
            self.seq_len = (input_size + window_size - 1) // window_size
            self.padded_size = self.seq_len * window_size
        else:
            self.seq_len = None
            self.padded_size = None

        self.lstm = nn.LSTM(
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
            padding = torch.zeros(batch_size, padded_size - actual_input_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif actual_input_size > padded_size:
            # Truncate if input is larger than expected
            x = x[:, :padded_size]
        
        # Reshape to sequence: (batch, seq_len, window_size)
        x = x.view(batch_size, seq_len, self.window_size)

        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        # c_n shape: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from the last layer
        # h_n[-1] shape: (batch, hidden_size)
        out = self.fc(h_n[-1])

        return out


if __name__ == "__main__":
    _ = LSTMNet(input_size=1000, output_size=2)
