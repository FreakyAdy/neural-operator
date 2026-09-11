"""Fourier-Attention Hybrid Operator.

Alternates between Fourier layers (global, long-range structure) and
local attention layers (fine-grained, local structure).

Motivation: pure FNO misses sharp local features; pure attention
is O(n²) in spatial resolution. The hybrid captures both scales.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from operatorlab.models.base import NeuralOperator
from operatorlab.models.fno import SpectralConv2d


class LocalAttention2d(nn.Module):
    """Local multi-head self-attention on 2D spatial fields.

    Operates on patches of the field to keep cost manageable.
    Uses standard multi-head attention within non-overlapping patches.

    Args:
        dim: Feature dimension.
        n_heads: Number of attention heads.
        patch_size: Size of local attention patches.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.patch_size = patch_size
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        """Apply local attention.

        Args:
            x: Input, shape (batch, channels, H, W).

        Returns:
            Output, shape (batch, channels, H, W).
        """
        b, c, h, w = x.shape
        ps = self.patch_size

        # Pad to multiple of patch_size if needed
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x.shape

        # Rearrange into patches: (batch, C, H, W) → (batch*n_patches, patch_size², C)
        n_h, n_w = hp // ps, wp // ps
        # (b, c, n_h, ps, n_w, ps)
        x = x.reshape(b, c, n_h, ps, n_w, ps)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (b, n_h, n_w, ps, ps, c)
        x = x.reshape(b * n_h * n_w, ps * ps, c)  # (B, seq_len, c)

        # Apply attention
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, seq, head_dim)

        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], c)
        out = self.proj(out) + residual

        # Reshape back: (b*n_h*n_w, ps*ps, c) → (b, c, hp, wp)
        out = out.reshape(b, n_h, n_w, ps, ps, c)
        out = out.permute(0, 5, 1, 3, 2, 4)  # (b, c, n_h, ps, n_w, ps)
        out = out.reshape(b, c, hp, wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out


class HybridOperator(NeuralOperator):
    """Fourier-Attention Hybrid Neural Operator.

    Alternates between:
        - Fourier layers: capture global, long-range structure
        - Local attention layers: capture fine-grained, local features

    Args:
        modes1: Fourier modes in first dimension.
        modes2: Fourier modes in second dimension.
        width: Channel width.
        n_fno_layers: Number of Fourier layers.
        n_attention_layers: Number of attention layers (interleaved).
        attention_heads: Number of attention heads.
        input_dim: Input channels.
        output_dim: Output channels.
        patch_size: Patch size for local attention.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_fno_layers: int = 4,
        n_attention_layers: int = 2,
        attention_heads: int = 4,
        input_dim: int = 1,
        output_dim: int = 1,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_fno_layers = n_fno_layers
        self.n_attention_layers = n_attention_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lifting = nn.Linear(input_dim + 2, width)

        # Interleave FNO and attention layers
        total_layers = n_fno_layers + n_attention_layers
        self.layer_types: list[str] = []
        self.spectral_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        # Distribute attention layers evenly among FNO layers
        attn_positions = set()
        if n_attention_layers > 0:
            step = total_layers / n_attention_layers
            for i in range(n_attention_layers):
                pos = int((i + 0.5) * step)
                attn_positions.add(min(pos, total_layers - 1))

        fno_idx = 0
        attn_idx = 0
        for i in range(total_layers):
            if i in attn_positions and attn_idx < n_attention_layers:
                self.layer_types.append("attention")
                self.attention_layers.append(
                    LocalAttention2d(width, n_heads=attention_heads, patch_size=patch_size)
                )
                # Still need skip conv for residual
                self.skip_convs.append(nn.Conv2d(width, width, kernel_size=1))
                attn_idx += 1
            else:
                self.layer_types.append("fourier")
                self.spectral_convs.append(
                    SpectralConv2d(width, width, modes1, modes2)
                )
                self.skip_convs.append(nn.Conv2d(width, width, kernel_size=1))
                fno_idx += 1

        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )
        self.activation = nn.GELU()

    def forward(self, a: Tensor, grid: Optional[Tensor] = None) -> Tensor:
        """Forward pass alternating Fourier and attention layers."""
        batch_size, h, w, _ = a.shape

        if grid is None:
            grid = self.make_grid((h, w), a.device)
            grid = grid.expand(batch_size, -1, -1, -1)

        x = torch.cat([a, grid], dim=-1)
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)

        fno_idx = 0
        attn_idx = 0
        for i, layer_type in enumerate(self.layer_types):
            skip = self.skip_convs[i](x)

            if layer_type == "fourier":
                x1 = self.spectral_convs[fno_idx](x)
                fno_idx += 1
            else:
                x1 = self.attention_layers[attn_idx](x)
                attn_idx += 1

            x = self.activation(x1 + skip)

        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def supports_resolution_transfer(self) -> bool:
        return True

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "modes1": self.modes1, "modes2": self.modes2,
            "width": self.width,
            "n_fno_layers": self.n_fno_layers,
            "n_attention_layers": self.n_attention_layers,
        })
        return config
