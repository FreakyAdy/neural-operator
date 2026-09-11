"""Fourier Neural Operator (FNO) for 2D problems.

Implements the spectral convolution layer and the full FNO2d architecture.
Based on: "Fourier Neural Operator for Parametric Partial Differential Equations"
(Li et al., 2020).

The FNO applies iterative kernel integration in the Fourier domain:
    v_{t+1}(x) = σ(W·v_t(x) + K(v_t)(x))
where K operates by pointwise multiplication of learnable weights R(k)
with the Fourier coefficients of v_t.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from operatorlab.models.base import NeuralOperator


def compl_mul2d(input_tensor: Tensor, weights: Tensor) -> Tensor:
    """Complex-valued batch matrix multiply for 2D spectral convolution.

    Computes einsum 'bixy,ioxy->boxy' for complex tensors.

    Args:
        input_tensor: Complex input, shape (batch, in_ch, H, W).
        weights: Complex weights, shape (in_ch, out_ch, H, W).

    Returns:
        Complex output, shape (batch, out_ch, H, W).
    """
    return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)


class SpectralConv2d(nn.Module):
    """Fourier-domain spectral convolution layer.

    Applies learnable pointwise multiplication in the Fourier domain:
        1. FFT of input
        2. Multiply by learnable complex weight matrices R(k)
        3. Inverse FFT

    Handles both positive and negative frequency halves of the spectrum
    to preserve conjugate symmetry.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Number of Fourier modes to keep in the first spatial dimension.
        modes2: Number of Fourier modes to keep in the second spatial dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Learnable complex weights for positive and negative frequency halves
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Args:
            x: Real-valued input, shape (batch, in_channels, H, W).

        Returns:
            Real-valued output, shape (batch, out_channels, H, W).
        """
        batch_size = x.shape[0]
        # Compute 2D real FFT: (batch, in_ch, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Allocate output Fourier tensor
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Effective modes bounded by available frequency bins
        m1 = min(self.modes1, x.size(-2) // 2)
        m2 = min(self.modes2, x.size(-1) // 2 + 1)

        # Multiply by weights at low-frequency modes (positive freqs in dim -2)
        out_ft[:, :, :m1, :m2] = compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        # Multiply by weights at low-frequency modes (negative freqs in dim -2)
        out_ft[:, :, -m1:, :m2] = compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        # Inverse FFT back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class FNO2d(NeuralOperator):
    """2D Fourier Neural Operator.

    Architecture:
        1. Lifting: pointwise MLP P maps input to width-dimensional space
        2. L Fourier layers: spectral conv + skip connection + activation
        3. Projection: pointwise MLP Q maps back to output dimension

    Grid coordinates (x, y) ∈ [0,1]² are always appended as input channels,
    making the actual lifting input dimension = input_dim + 2.

    The model is resolution-independent: spectral truncation at k_max modes
    means the same weights work at any spatial resolution.

    Args:
        modes1: Number of Fourier modes in the first spatial dimension.
        modes2: Number of Fourier modes in the second spatial dimension.
        width: Channel width in the lifted representation.
        n_layers: Number of Fourier integral operator layers.
        input_dim: Number of input channels (excluding grid coordinates).
        output_dim: Number of output channels.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        input_dim: int = 1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Lifting: input_dim + 2 (grid coords) → width
        self.lifting = nn.Linear(input_dim + 2, width)

        # Fourier layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2)
            for _ in range(n_layers)
        ])
        # Residual skip connections (pointwise 1x1 conv)
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])

        # Projection: width → 128 → output_dim
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

        self.activation = nn.GELU()

    def forward(
        self,
        a: Tensor,
        grid: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the FNO.

        Args:
            a: Input function values, shape (batch, H, W, input_dim).
            grid: Spatial coordinates, shape (batch, H, W, 2).
                  If None, a uniform [0,1]² grid is constructed.

        Returns:
            Output function values, shape (batch, H, W, output_dim).
        """
        batch_size, h, w, _ = a.shape

        # Construct grid if not provided
        if grid is None:
            grid = self.make_grid((h, w), a.device)  # (1, H, W, 2)
            grid = grid.expand(batch_size, -1, -1, -1)  # (batch, H, W, 2)

        # Append grid coordinates to input: (batch, H, W, input_dim + 2)
        x = torch.cat([a, grid], dim=-1)

        # Lifting: (batch, H, W, input_dim + 2) → (batch, H, W, width)
        x = self.lifting(x)

        # Permute to channels-first for convolutions: (batch, width, H, W)
        x = x.permute(0, 3, 1, 2)

        # Fourier layers with residual skip connections
        for spectral_conv, skip_conv in zip(self.spectral_convs, self.skip_convs):
            x1 = spectral_conv(x)   # (batch, width, H, W)
            x2 = skip_conv(x)       # (batch, width, H, W)
            x = self.activation(x1 + x2)

        # Permute back: (batch, width, H, W) → (batch, H, W, width)
        x = x.permute(0, 2, 3, 1)

        # Projection: (batch, H, W, width) → (batch, H, W, output_dim)
        x = self.projection(x)

        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def supports_resolution_transfer(self) -> bool:
        """FNO supports zero-shot resolution transfer via spectral truncation."""
        return True

    def get_config(self) -> dict:
        """Return FNO hyperparameters for experiment tracking."""
        config = super().get_config()
        config.update({
            "modes1": self.modes1,
            "modes2": self.modes2,
            "width": self.width,
            "n_layers": self.n_layers,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config
