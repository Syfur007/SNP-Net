import torch
from torch import nn


class DenseNet(nn.Module):
    """A fully-connected neural network for SNP classification."""

    def __init__(
        self,
        hidden_sizes: list = None,
        output_size: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        input_size: int = None,  # Optional, for backward compatibility
    ) -> None:
        """Initialize a `DenseNet` module.

        :param hidden_sizes: List of hidden layer sizes. Defaults to [512, 256, 128].
        :param output_size: The number of output classes.
        :param dropout: Dropout probability.
        :param use_batch_norm: Whether to use batch normalization.
        :param input_size: Optional. The number of input features. If None, will be inferred from data.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self._input_size = input_size
        self.model = None
        
        # If input_size is provided, build the model immediately
        if input_size is not None:
            self._build_model(input_size)

    def _build_model(self, input_size: int) -> None:
        """Build the model layers once input size is known."""
        layers = []
        prev_size = input_size

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))

        self.model = nn.Sequential(*layers)
        self._input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor of shape (batch, num_snps).
        :return: A tensor of predictions.
        """
        # Build model on first forward pass if not already built
        if self.model is None:
            input_size = x.size(1)
            self._build_model(input_size)
            # Move model to the same device as input
            self.model = self.model.to(x.device)
        
        # Verify input size matches expected size
        expected_size = self._input_size
        actual_size = x.size(1)
        if expected_size != actual_size:
            raise ValueError(
                f"Input size mismatch: expected {expected_size} features, "
                f"but got {actual_size} features. "
                f"The model was built for {expected_size} features based on the first batch."
            )
        
        return self.model(x)


if __name__ == "__main__":
    _ = DenseNet(input_size=1000, output_size=2)
