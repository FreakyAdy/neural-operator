"""Scientific validity and physical integrity subsystem."""

from __future__ import annotations

from operatorlab.validity.conservation import ConservationAuditResult, audit_conservation
from operatorlab.validity.invariants import InvariantsAuditResult, audit_invariants
from operatorlab.validity.spectral import SpectralAuditResult, audit_spectral_fidelity
from operatorlab.validity.stability import StabilityAuditResult, audit_stability

__all__ = [
    "ConservationAuditResult",
    "InvariantsAuditResult",
    "SpectralAuditResult",
    "StabilityAuditResult",
    "audit_conservation",
    "audit_invariants",
    "audit_spectral_fidelity",
    "audit_stability",
]
