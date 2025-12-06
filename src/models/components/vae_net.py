"""Variational Autoencoder (VAE) model for SNP classification.

This module implements a Variational Autoencoder that learns a probabilistic
latent representation of SNP data and uses it for classification.
"""

import torch
from torch import nn


class VAENet(nn.Module):
    """Variational Autoencoder network for SNP classification.

    The model consists of:
    1. Encoder: Maps input to latent distribution parameters (mu, logvar)
    2. Reparameterization: Samples from latent distribution
    3. Classifier: Maps latent samples to class predictions
    4. (Optional) Decoder: Reconstructs input from latent samples

    The decoder is included for potential future use with reconstruction loss
    and KL divergence, but the current implementation focuses on classification only.
    """

    def __init__(
        self,
        hidden_sizes: list[int] = None,
        latent_dim: int = 64,
        output_size: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        input_size: int = None,
    ) -> None:
        """Initialize VAENet.

        :param hidden_sizes: List of hidden layer sizes for encoder (e.g., [512, 256, 128]).
            Decoder will mirror this structure in reverse. Default: [512, 256, 128].
        :param latent_dim: Dimension of the latent (bottleneck) representation.
        :param output_size: Number of output classes for classification.
        :param dropout: Dropout probability for regularization.
        :param use_batch_norm: Whether to use batch normalization in layers.
        :param input_size: Optional input feature size. If None, model builds on first forward pass.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        self._input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Will be built dynamically on first forward pass if input_size not provided
        self.encoder = None
        self.fc_mu = None
        self.fc_logvar = None
        self.decoder = None
        self.classifier = None

        if input_size is not None:
            self._build_model(input_size)

    def _build_encoder(self, input_size: int) -> nn.Sequential:
        """Build the encoder network (up to the point before mu/logvar split).

        :param input_size: Number of input features.
        :return: Sequential encoder module.
        """
        layers = []
        in_features = input_size

        # Build encoder layers
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_features = hidden_size

        return nn.Sequential(*layers)

    def _build_decoder(self, output_size: int) -> nn.Sequential:
        """Build the decoder network (mirrors encoder structure).

        :param output_size: Number of output features (should match input_size).
        :return: Sequential decoder module.
        """
        layers = []
        in_features = self.latent_dim

        # Build decoder layers (reverse of encoder)
        for hidden_size in reversed(self.hidden_sizes):
            layers.append(nn.Linear(in_features, hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_features = hidden_size

        # Final layer to reconstruct input
        layers.append(nn.Linear(in_features, output_size))

        return nn.Sequential(*layers)

    def _build_classifier(self) -> nn.Sequential:
        """Build the classifier network from latent space to class predictions.

        :return: Sequential classifier module.
        """
        return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim // 2, self.output_size),
        )

    def _build_model(self, input_size: int) -> None:
        """Build all model components.

        :param input_size: Number of input features.
        """
        self.encoder = self._build_encoder(input_size)

        # Latent distribution parameters
        encoder_output_dim = self.hidden_sizes[-1]
        self.fc_mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, self.latent_dim)

        # Decoder and classifier
        self.decoder = self._build_decoder(input_size)
        self.classifier = self._build_classifier()

        self._input_size = input_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        :param x: Input tensor of shape (batch_size, num_features).
        :return: Tuple of (mu, logvar) each of shape (batch_size, latent_dim).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from N(mu, var) using N(0,1).

        z = mu + sigma * epsilon, where epsilon ~ N(0,1)
        This allows gradients to flow through the sampling operation.

        :param mu: Mean of latent distribution, shape (batch_size, latent_dim).
        :param logvar: Log variance of latent distribution, shape (batch_size, latent_dim).
        :return: Sampled latent vector of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space.

        :param z: Latent tensor of shape (batch_size, latent_dim).
        :return: Reconstructed input of shape (batch_size, num_features).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.

        During training, uses reparameterization to sample from latent distribution.
        During inference, can use either mu directly or sampled values.

        :param x: Input tensor of shape (batch_size, num_features).
        :return: Classification logits of shape (batch_size, output_size).
        """
        # Build model on first forward pass if needed
        if self.encoder is None:
            input_size = x.size(1)
            self._build_model(input_size)
            # Move to same device as input
            self.encoder = self.encoder.to(x.device)
            self.fc_mu = self.fc_mu.to(x.device)
            self.fc_logvar = self.fc_logvar.to(x.device)
            self.decoder = self.decoder.to(x.device)
            self.classifier = self.classifier.to(x.device)

        # Encode to latent distribution
        mu, logvar = self.encode(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Classify from latent sample
        logits = self.classifier(z)

        return logits

    def forward_with_reconstruction(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with classification, reconstruction, and distribution parameters.

        This method is provided for potential future use with reconstruction loss
        and KL divergence regularization.

        :param x: Input tensor of shape (batch_size, num_features).
        :return: Tuple of (logits, reconstruction, mu, logvar).
        """
        # Build model on first forward pass if needed
        if self.encoder is None:
            input_size = x.size(1)
            self._build_model(input_size)
            self.encoder = self.encoder.to(x.device)
            self.fc_mu = self.fc_mu.to(x.device)
            self.fc_logvar = self.fc_logvar.to(x.device)
            self.decoder = self.decoder.to(x.device)
            self.classifier = self.classifier.to(x.device)

        # Encode to latent distribution
        mu, logvar = self.encode(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Classify from latent sample
        logits = self.classifier(z)

        # Reconstruct input
        reconstruction = self.decode(z)

        return logits, reconstruction, mu, logvar

    def get_latent_representation(
        self, x: torch.Tensor, use_mean: bool = True
    ) -> torch.Tensor:
        """Get latent representation of input.

        :param x: Input tensor of shape (batch_size, num_features).
        :param use_mean: If True, return mu (mean of distribution).
            If False, sample from distribution using reparameterization.
        :return: Latent representation of shape (batch_size, latent_dim).
        """
        if self.encoder is None:
            input_size = x.size(1)
            self._build_model(input_size)
            self.encoder = self.encoder.to(x.device)
            self.fc_mu = self.fc_mu.to(x.device)
            self.fc_logvar = self.fc_logvar.to(x.device)

        mu, logvar = self.encode(x)

        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)


if __name__ == "__main__":
    # Test the model
    print("Testing VAENet...")

    # Test with dynamic input sizing
    model = VAENet(
        hidden_sizes=[512, 256, 128], latent_dim=64, output_size=2, dropout=0.3
    )

    # Simulate SNP data
    batch_size = 32
    num_snps = 1000
    x = torch.randn(batch_size, num_snps)

    print(f"Input shape: {x.shape}")

    # Test classification forward pass
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 2), "Incorrect output shape"

    # Test forward with reconstruction
    logits, reconstruction, mu, logvar = model.forward_with_reconstruction(x)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    assert reconstruction.shape == x.shape, "Incorrect reconstruction shape"
    assert mu.shape == (batch_size, 64), "Incorrect mu shape"
    assert logvar.shape == (batch_size, 64), "Incorrect logvar shape"

    # Test latent representation extraction
    latent_mean = model.get_latent_representation(x, use_mean=True)
    latent_sample = model.get_latent_representation(x, use_mean=False)
    print(f"Latent mean shape: {latent_mean.shape}")
    print(f"Latent sample shape: {latent_sample.shape}")
    assert latent_mean.shape == (batch_size, 64), "Incorrect latent shape"
    assert latent_sample.shape == (batch_size, 64), "Incorrect latent shape"

    # Test with pre-specified input size
    model2 = VAENet(
        hidden_sizes=[256, 128], latent_dim=32, output_size=3, input_size=num_snps
    )
    logits2 = model2(x)
    print(f"Pre-built model output shape: {logits2.shape}")
    assert logits2.shape == (batch_size, 3), "Incorrect output shape for pre-built model"

    # Test reparameterization consistency
    mu_test = torch.zeros(10, 32)
    logvar_test = torch.zeros(10, 32)
    z1 = model.reparameterize(mu_test, logvar_test)
    z2 = model.reparameterize(mu_test, logvar_test)
    print(f"Reparameterization test - Different samples: {not torch.allclose(z1, z2)}")

    print("\n✓ All tests passed!")
