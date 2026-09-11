"""Unit tests for the operatorlab.validity subsystem."""

from __future__ import annotations

import pytest
import torch

from operatorlab.validity.conservation import audit_conservation
from operatorlab.validity.invariants import audit_invariants
from operatorlab.validity.spectral import audit_spectral_fidelity
from operatorlab.validity.stability import audit_stability


class TestValiditySubsystem:
    def test_conservation_audit(self) -> None:
        init = torch.ones(2, 16, 16, 1) * 2.0
        target = init.clone()
        pred = init * 1.02  # 2% error
        res = audit_conservation(pred, init, target)
        assert res.mass_error_percent >= 0.0
        assert res.energy_drift_percent >= 0.0
        assert res.mass_status in ["PASS", "WARN", "FAIL"]

    def test_stability_audit(self) -> None:
        ref = torch.ones(2, 16, 16, 1)
        pred = ref * 1.5
        res = audit_stability(pred, ref, max_allowed_growth=3.0)
        assert res.is_finite is True
        assert res.status == "PASS"

        # Test NaN blowup
        nan_pred = torch.full((2, 16, 16, 1), float("nan"))
        nan_res = audit_stability(nan_pred, ref)
        assert nan_res.is_finite is False
        assert nan_res.status == "FAIL"

    def test_invariants_audit(self) -> None:
        pred = torch.randn(2, 16, 16, 1)
        res = audit_invariants(pred, device="cpu")
        assert res.divergence_norm >= 0.0
        assert res.divergence_status in ["PASS", "WARN", "FAIL"]

    def test_spectral_audit(self) -> None:
        target = torch.randn(2, 16, 16, 1)
        pred = target.clone()
        res = audit_spectral_fidelity(pred, target)
        assert res.spectrum_error < 0.01
        assert res.spectrum_status == "PASS"
