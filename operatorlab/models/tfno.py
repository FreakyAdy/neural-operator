"""Tensorized Fourier Neural Operator (TFNO).

Derives from FNO by replacing dense spectral weight tensors R(k) with
Tucker-decomposed factors, reducing parameter count by ~10x at minimal
accuracy cost.

Reference: Kossaifi et al., "Multi-Grid Tensorized Fourier Neural Operator
for High-Resolution PDEs" (2023).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from operatorlab.models.base import NeuralOperator
from operatorlab.models.fno import compl_mul2d


class TuckerSpectralConv2d(nn.Module):
    """Tucker-decomposed spectral convolution layer.

    Replaces the dense weight tensor R of shape (in_ch, out_ch, modes1, modes2)
    with Tucker factors: core tensor and factor matrices.

    This reduces parameter count from O(in*out*m1*m2) to
    O(r*(in + out + m1 + m2) + r^4) where r is the rank.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Fourier modes in first dimension.
        modes2: Fourier modes in second dimension.
        rank: Tucker decomposition rank.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank

        scale = 1.0 / (in_channels * out_channels)

        # Tucker factors for positive frequencies
        self.core1 = nn.Parameter(scale * torch.randn(rank, rank, rank, rank, dtype=torch.cfloat))
        self.factor_in1 = nn.Parameter(scale * torch.randn(in_channels, rank, dtype=torch.cfloat))
        self.factor_out1 = nn.Parameter(scale * torch.randn(out_channels, rank, dtype=torch.cfloat))
        self.factor_m1_1 = nn.Parameter(scale * torch.randn(modes1, rank, dtype=torch.cfloat))
        self.factor_m2_1 = nn.Parameter(scale * torch.randn(modes2, rank, dtype=torch.cfloat))

        # Tucker factors for negative frequencies
        self.core2 = nn.Parameter(scale * torch.randn(rank, rank, rank, rank, dtype=torch.cfloat))
        self.factor_in2 = nn.Parameter(scale * torch.randn(in_channels, rank, dtype=torch.cfloat))
        self.factor_out2 = nn.Parameter(scale * torch.randn(out_channels, rank, dtype=torch.cfloat))
        self.factor_m1_2 = nn.Parameter(scale * torch.randn(modes1, rank, dtype=torch.cfloat))
        self.factor_m2_2 = nn.Parameter(scale * torch.randn(modes2, rank, dtype=torch.cfloat))

    def _reconstruct_weights(
        self, core: Tensor, f_in: Tensor, f_out: Tensor, f_m1: Tensor, f_m2: Tensor,
    ) -> Tensor:
        """Reconstruct full weight tensor from Tucker factors.

        core: (r, r, r, r)
        f_in: (in_ch, r), f_out: (out_ch, r), f_m1: (m1, r), f_m2: (m2, r)
        Returns: (in_ch, out_ch, m1, m2)
        """
        # Contract: W = core ×₁ f_in ×₂ f_out ×₃ f_m1 ×₄ f_m2
        w = torch.einsum("abcd,ia,ob,xc,yd->ioxy", core, f_in, f_out, f_m1, f_m2)
        return w

    def forward(self, x: Tensor) -> Tensor:
        """Apply Tucker-decomposed spectral convolution.

        Args:
            x: Input, shape (batch, in_channels, H, W).

        Returns:
            Output, shape (batch, out_channels, H, W).
        """
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # Reconstruct weights and apply
        w1 = self._reconstruct_weights(
            self.core1, self.factor_in1, self.factor_out1,
            self.factor_m1_1, self.factor_m2_1,
        )
        w2 = self._reconstruct_weights(
            self.core2, self.factor_in2, self.factor_out2,
            self.factor_m1_2, self.factor_m2_2,
        )

        # Effective modes bounded by available frequency bins
        m1 = min(self.modes1, x.size(-2) // 2)
        m2 = min(self.modes2, x.size(-1) // 2 + 1)

        out_ft[:, :, :m1, :m2] = compl_mul2d(
            x_ft[:, :, :m1, :m2], w1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = compl_mul2d(
            x_ft[:, :, -m1:, :m2], w2[:, :, :m1, :m2],
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")


class TFNO2d(NeuralOperator):
    """2D Tensorized Fourier Neural Operator.

    Same architecture as FNO2d but with Tucker-decomposed spectral weights.
    Reduces parameter count by ~10x at minimal accuracy cost.

    Args:
        modes1: Fourier modes in first dimension.
        modes2: Fourier modes in second dimension.
        width: Channel width.
        n_layers: Number of Fourier layers.
        input_dim: Input channels (excluding grid coords).
        output_dim: Output channels.
        rank: Tucker decomposition rank.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        input_dim: int = 1,
        output_dim: int = 1,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.lifting = nn.Linear(input_dim + 2, width)

        self.spectral_convs = nn.ModuleList([
            TuckerSpectralConv2d(width, width, modes1, modes2, rank)
            for _ in range(n_layers)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )
        self.activation = nn.GELU()

    def forward(self, a: Tensor, grid: Optional[Tensor] = None) -> Tensor:
        """Forward pass — identical to FNO2d but with Tucker spectral layers."""
        batch_size, h, w, _ = a.shape

        if grid is None:
            grid = self.make_grid((h, w), a.device)
            grid = grid.expand(batch_size, -1, -1, -1)

        x = torch.cat([a, grid], dim=-1)        # (batch, H, W, in_dim+2)
        x = self.lifting(x)                      # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)               # (batch, width, H, W)

        for spectral_conv, skip_conv in zip(self.spectral_convs, self.skip_convs):
            x1 = spectral_conv(x)
            x2 = skip_conv(x)
            x = self.activation(x1 + x2)

        x = x.permute(0, 2, 3, 1)               # (batch, H, W, width)
        x = self.projection(x)                   # (batch, H, W, out_dim)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def supports_resolution_transfer(self) -> bool:
        return True

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "modes1": self.modes1, "modes2": self.modes2,
            "width": self.width, "n_layers": self.n_layers,
            "rank": self.rank,
        })
        return config
