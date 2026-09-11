"""Unit tests for loss functions."""

from __future__ import annotations

import torch
import pytest

from operatorlab.training.losses import relative_l2, h1_loss, OperatorLoss


class TestRelativeL2:
    """Tests for the relative L2 loss."""

    def test_perfect_prediction(self) -> None:
        """Relative L2 of identical tensors should be zero."""
        pred = torch.ones(4, 32, 32, 1)
        target = torch.ones(4, 32, 32, 1)
        loss = relative_l2(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive(self) -> None:
        """Relative L2 should be positive for different tensors."""
        pred = torch.randn(4, 32, 32, 1)
        target = torch.randn(4, 32, 32, 1)
        loss = relative_l2(pred, target)
        assert loss.item() > 0

    def test_scalar_output(self) -> None:
        """Relative L2 returns a scalar tensor."""
        pred = torch.randn(8, 16, 16, 1)
        target = torch.randn(8, 16, 16, 1)
        loss = relative_l2(pred, target)
        assert loss.ndim == 0

    def test_near_zero_target(self) -> None:
        """Relative L2 handles near-zero targets without NaN (epsilon guard)."""
        pred = torch.randn(4, 16, 16, 1)
        target = torch.zeros(4, 16, 16, 1)
        loss = relative_l2(pred, target)
        assert not torch.isnan(loss), "Loss should not be NaN for zero target"
        assert not torch.isinf(loss), "Loss should not be Inf for zero target"


class TestH1Loss:
    """Tests for the H1 Sobolev loss."""

    def test_perfect_prediction(self) -> None:
        """H1 loss of identical tensors should be near zero."""
        pred = torch.ones(4, 32, 32, 1)
        target = torch.ones(4, 32, 32, 1)
        loss = h1_loss(pred, target, dx=1.0 / 32)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive(self) -> None:
        """H1 loss should be positive for different tensors."""
        pred = torch.randn(4, 32, 32, 1)
        target = torch.randn(4, 32, 32, 1)
        loss = h1_loss(pred, target, dx=1.0 / 32)
        assert loss.item() > 0

    def test_greater_than_l2(self) -> None:
        """H1 loss should be >= relative L2 (it adds gradient penalty)."""
        pred = torch.randn(4, 32, 32, 1)
        target = torch.randn(4, 32, 32, 1)
        l2 = relative_l2(pred, target)
        h1 = h1_loss(pred, target, dx=1.0 / 32)
        assert h1.item() >= l2.item() - 1e-6


class TestOperatorLoss:
    """Tests for the combined OperatorLoss module."""

    def test_l2_only(self) -> None:
        """OperatorLoss with only L2 matches relative_l2."""
        loss_fn = OperatorLoss(l2_weight=1.0, h1_weight=0.0, pde_weight=0.0)
        pred = torch.randn(4, 32, 32, 1)
        target = torch.randn(4, 32, 32, 1)
        combined = loss_fn(pred, target)
        direct = relative_l2(pred, target)
        assert combined.item() == pytest.approx(direct.item(), rel=1e-5)

    def test_h1_adds_to_loss(self) -> None:
        """Adding H1 weight increases the loss (for non-identical tensors)."""
        pred = torch.randn(4, 32, 32, 1)
        target = torch.randn(4, 32, 32, 1)

        l2_only = OperatorLoss(l2_weight=1.0, h1_weight=0.0)(pred, target)
        with_h1 = OperatorLoss(l2_weight=1.0, h1_weight=0.1)(pred, target)
        assert with_h1.item() > l2_only.item()

    def test_pde_weight_without_pde_raises(self) -> None:
        """Setting pde_weight > 0 without a PDE raises ValueError."""
        with pytest.raises(ValueError, match="pde must be provided"):
            OperatorLoss(pde_weight=0.1, pde=None)
