"""Per-sensor 1-D CNN models: DeepSVDD, ConvVAE, and a PCA baseline.

All neural models share a sensor-agnostic architecture: a single window of one
sensor is a (batch, 1, 60) sequence. Multivariate windows are scored by folding
the channel axis into the batch axis, so every sensor uses the same weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import nn

Array = NDArray[Any]

_EPS = 0.1


def build_trunk(width: int) -> nn.Sequential:
    """Shared bias-free temporal feature extractor."""
    return nn.Sequential(
        nn.Conv1d(1, width, kernel_size=5, stride=2, padding=2, bias=False),
        nn.LeakyReLU(0.1),
        nn.Conv1d(width, width * 2, kernel_size=5, stride=2, padding=2, bias=False),
        nn.LeakyReLU(0.1),
        nn.Conv1d(width * 2, width * 4, kernel_size=5, stride=2, padding=2, bias=False),
        nn.LeakyReLU(0.1),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


def build_encoder(width: int, latent_dim: int) -> nn.Sequential:
    """Shared trunk followed by a bias-free latent projection."""
    return nn.Sequential(build_trunk(width), nn.Linear(width * 4, latent_dim, bias=False))


def build_decoder(width: int, latent_dim: int, window: int) -> nn.Sequential:
    """Mirror of the encoder: linear lift + three stride-2 transposed convolutions."""

    class Crop(nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return z[..., :window]


    class Reshape(nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return z.view(z.size(0), -1, 1)

    return nn.Sequential(
        nn.Linear(latent_dim, width * 4),
        Reshape(),
        nn.Upsample(size=int(np.ceil(window / 8)), mode="nearest"),
        nn.ConvTranspose1d(width * 4, width * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1),
        nn.ConvTranspose1d(width * 2, width, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1),
        nn.ConvTranspose1d(width, 1, kernel_size=4, stride=2, padding=1, bias=False),
        Crop(),
    )


class DeepSVDD(nn.Module):
    """Deep SVDD over per-sensor windows; the center is a non-trainable buffer."""


    def __init__(self, width: int = 16, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = build_encoder(width, latent_dim)
        self.register_buffer("center", torch.zeros(latent_dim))
        self.center: torch.Tensor = self._buffers["center"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def initialize_center(self, windows: torch.Tensor, batch_size: int = 4096) -> None:
        """Official initialization: one forward pass, mean latent, push near-zero

        coordinates away from zero to |c| = 0.1 (avoids trivial hypersphere collapse).
        """
        device = self.center.device
        total = torch.zeros_like(self.center)
        count = 0
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size].to(device)
            total += self.encoder(batch).sum(dim=0)
            count += len(batch)
        center = total / max(count, 1)
        center[(center.abs() < _EPS) & (center > 0)] = _EPS
        center[(center.abs() < _EPS) & (center < 0)] = -_EPS
        self.center.copy_(center)


class ConvVAE(nn.Module):
    """Convolutional VAE reusing the shared encoder; scoring uses mu only."""

    def __init__(self, width: int = 16, latent_dim: int = 16, window: int = 60) -> None:
        super().__init__()
        self.trunk = build_trunk(width)
        self.mu_head = nn.Linear(width * 4, latent_dim)
        self.logvar_head = nn.Linear(width * 4, latent_dim)
        self.decoder = build_decoder(width, latent_dim, window)
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.mu_head(features), self.logvar_head(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return self.decoder(z), mu, logvar


def _channel_batches(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """(B, C, T) -> ((B*C, 1, T), (B, C)); fold sensors into the batch axis."""
    batch, channels, time = x.shape
    return x.permute(0, 1, 2).reshape(batch * channels, 1, time), (batch, channels)


@torch.no_grad()
def svdd_sensor_scores(model: DeepSVDD, x: torch.Tensor) -> torch.Tensor:
    """Squared latent distance to the center per sensor: (B, C, T) -> (B, C)."""
    flat, (batch, channels) = _channel_batches(x)
    z = model.encoder(flat.to(model.center.device))
    scores = torch.sum((z - model.center) ** 2, dim=1)
    return scores.view(batch, channels).cpu()


@torch.no_grad()
def vae_sensor_scores(model: ConvVAE, x: torch.Tensor) -> torch.Tensor:
    """0.5 * MSE per sensor (unit-variance Gaussian NLL up to a constant): (B, C, T) -> (B, C).

    Scores use mu only — never a reparameterized sample.
    """
    flat, (batch, channels) = _channel_batches(x)
    device = next(model.parameters()).device
    flat = flat.to(device)
    mu, _ = model.encode(flat)
    reconstruction = model.decoder(mu)
    scores = 0.5 * torch.mean((flat - reconstruction) ** 2, dim=(1, 2))
    return scores.view(batch, channels).cpu()


class PCAReconstructionBaseline:
    """Randomized PCA over flattened non-overlapping multivariate windows.

    Scores are 0.5 * per-sensor MSE after inverse_transform.
    """

    def __init__(self, n_components: int, random_state: int) -> None:
        self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)

    def fit(self, windows: Array) -> "PCAReconstructionBaseline":
        self.pca.fit(self._flatten(windows))
        return self

    def sensor_scores(self, windows: Array) -> Array:
        flat = self._flatten(windows)
        reconstruction = self.pca.inverse_transform(self.pca.transform(flat))
        squared = (flat - reconstruction) ** 2
        n_channels = windows.shape[1]
        return 0.5 * squared.reshape(len(flat), n_channels, -1).mean(axis=2)

    @staticmethod
    def _flatten(windows: Array) -> Array:
        """(N, C, T) -> (N, C*T)."""
        return windows.reshape(len(windows), -1).astype(np.float64, copy=False)
