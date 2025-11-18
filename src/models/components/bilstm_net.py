import torch
from torch import nn


class BiLSTMNet(nn.Module):
    """A Bidirectional LSTM-based neural network for sequence classification."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 10,
        dropout: float = 0.2,
    ) -> None:
        """Initialize a `BiLSTMNet` module.

        :param input_size: The number of input features per timestep.
        :param hidden_size: The number of features in the hidden state (per direction).
        :param num_layers: Number of recurrent layers.
        :param output_size: The number of output features (classes).
        :param dropout: Dropout probability between LSTM layers.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Bidirectional LSTM produces hidden_size * 2 output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor of shape (batch, channels, height, width).
        :return: A tensor of predictions of shape (batch, output_size).
        """
        batch_size, channels, height, width = x.size()

        # Reshape image to sequence: (batch, height, width) treating height as sequence length
        # For MNIST: (batch, 1, 28, 28) -> (batch, 28, 28)
        x = x.view(batch_size, height, width)

        # BiLSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # c_n shape: (num_layers * 2, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.bilstm(x)

        # Concatenate the final forward and backward hidden states
        # h_n[-2] is the last layer's forward direction
        # h_n[-1] is the last layer's backward direction
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat((h_forward, h_backward), dim=1)

        out = self.fc(h_concat)

        return out


if __name__ == "__main__":
    _ = BiLSTMNet()
