"""Visualization: field plots, spectra, and temporal rollout animations."""

from __future__ import annotations

import matplotlib

# Force headless Agg backend to prevent Tkinter initialization errors
matplotlib.use("Agg")

from operatorlab.visualization.fields import plot_field_comparison
from operatorlab.visualization.spectra import plot_error_spectrum
from operatorlab.visualization.trajectories import animate_rollout

__all__ = [
    "plot_field_comparison",
    "plot_error_spectrum",
    "animate_rollout",
]
