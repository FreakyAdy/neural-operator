"""Unit tests for dataset transformations and normalizers."""

from __future__ import annotations

import pytest
import torch

from operatorlab.data.datasets import InMemoryDataset
from operatorlab.data.transforms import UnitGaussianNormalizer


class TestUnitGaussianNormalizer:
    def test_fit_and_encode(self) -> None:
        normalizer = UnitGaussianNormalizer()
        x = torch.randn(10, 16, 16, 2) * 5.0 + 3.0
        normalizer.fit(x)
        encoded = normalizer.encode(x)
        # Mean across samples and spatial dims should be near 0, std near 1
        assert torch.allclose(encoded.mean(dim=(0, 1, 2)), torch.zeros(2), atol=1e-4)
        assert torch.allclose(encoded.std(dim=(0, 1, 2)), torch.ones(2), atol=1e-3)

    def test_decode_roundtrip(self) -> None:
        normalizer = UnitGaussianNormalizer()
        x = torch.randn(5, 8, 8, 1) * 2.0 + 1.0
        normalizer.fit(x)
        encoded = normalizer.encode(x)
        decoded = normalizer.decode(encoded)
        assert torch.allclose(decoded, x, atol=1e-5)

    def test_state_dict_serialization(self) -> None:
        norm1 = UnitGaussianNormalizer()
        x = torch.randn(4, 8, 8, 1)
        norm1.fit(x)
        state = norm1.state_dict()

        norm2 = UnitGaussianNormalizer()
        norm2.load_state_dict(state)
        assert torch.allclose(norm1.mean, norm2.mean)
        assert torch.allclose(norm1.std, norm2.std)


class TestInMemoryDataset:
    def test_item_access(self) -> None:
        a = torch.randn(5, 16, 16, 1)
        u = torch.randn(5, 16, 16, 1)
        ds = InMemoryDataset(a, u, normalize=True)
        assert len(ds) == 5
        sample_a, sample_u, grid = ds[0]
        assert sample_a.shape == (16, 16, 1)
        assert sample_u.shape == (16, 16, 1)
        assert grid.shape == (16, 16, 2)
