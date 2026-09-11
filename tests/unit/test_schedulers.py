"""Unit tests for learning rate scheduler builder."""

from __future__ import annotations

import pytest
import torch

from operatorlab.training.schedulers import build_scheduler


class TestSchedulers:
    def test_build_cosine(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        sched = build_scheduler(opt, "cosine", epochs=100)
        assert sched is not None

    def test_build_step(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        sched = build_scheduler(opt, "step", epochs=100)
        assert sched is not None

    def test_build_plateau(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        sched = build_scheduler(opt, "plateau", epochs=100)
        assert sched is not None

    def test_build_none(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        sched = build_scheduler(opt, "none", epochs=100)
        assert sched is None

    def test_invalid_scheduler(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        with pytest.raises(ValueError):
            build_scheduler(opt, "non_existent", epochs=100)
