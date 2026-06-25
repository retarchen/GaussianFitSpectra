"""Lightweight 1D Gaussian spectral decomposition.

The public API is intentionally small. Most users will read a three-column
spectrum table with :func:`read_spectrum`, fit it with :func:`fit_spectrum`,
save the returned component table, and optionally create a diagnostic plot with
:func:`plot_fit`.
"""

from .fitting import SpectrumFitResult, fit_spectrum
from .io import read_spectrum, save_components_csv
from .plotting import plot_fit

__all__ = [
    "SpectrumFitResult",
    "fit_spectrum",
    "plot_fit",
    "read_spectrum",
    "save_components_csv",
]
