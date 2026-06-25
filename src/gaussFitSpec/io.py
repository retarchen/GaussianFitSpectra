"""Input and output helpers for gaussFitSpec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_spectrum(path):
    """Read a plain table containing one 1D spectrum.

    The input file must contain exactly three numerical columns in this order:
    velocity, spectrum, and spectrum uncertainty. Both whitespace-separated and
    comma-separated files are accepted, and lines beginning with ``#`` are
    ignored.

    Args:
        path: Path to a ``.txt``, ``.dat``, ``.csv``, or similar plain text
            table.

    Returns:
        A tuple ``(velocity, spectrum, spectrum_err)`` of NumPy arrays.

    Raises:
        ValueError: If the file does not contain exactly three columns or if any
            value cannot be converted to a floating-point number.

    Example:
        .. code-block:: python

           from gaussFitSpec import read_spectrum

           velocity, spectrum, spectrum_err = read_spectrum("examples/example_spectra.txt")
    """

    file_path = Path(path)
    frame = pd.read_csv(
        file_path,
        sep=r"[\s,]+",
        engine="python",
        header=None,
        comment="#",
    )
    frame = frame.dropna(how="all", axis=1)
    if frame.shape[1] != 3:
        raise ValueError(
            f"{file_path} must contain exactly three numerical columns: "
            "velocity, spectrum, spectrum_err."
        )

    try:
        numeric = frame.astype(float)
    except ValueError as exc:
        raise ValueError(f"{file_path} contains non-numerical values.") from exc

    return tuple(numeric[column].to_numpy(dtype=float) for column in numeric.columns)


def save_components_csv(result, path, index=False):
    """Save the fitted Gaussian component table to CSV.

    This is a small convenience wrapper around
    ``result.components.to_csv(...)``. It accepts the
    :class:`gaussFitSpec.fitting.SpectrumFitResult` returned by
    :func:`gaussFitSpec.fit_spectrum`.

    Args:
        result: Fit result returned by :func:`gaussFitSpec.fit_spectrum`.
        path: Output CSV file path.
        index: Passed to :meth:`pandas.DataFrame.to_csv`. The default ``False``
            avoids writing the DataFrame index.

    Example:
        .. code-block:: python

           from gaussFitSpec import save_components_csv

           save_components_csv(result, "examples/example_components.csv")
    """

    result.components.to_csv(path, index=index)
