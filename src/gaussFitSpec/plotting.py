"""Plotting helpers for gaussFitSpec."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_fit(
    velocity,
    spectrum,
    spectrum_err,
    result,
    *,
    output_file=None,
    show=False,
    figure=None,
    axes=None,
):
    """Create a diagnostic plot for a fitted spectrum.

    The figure contains an upper panel with the observed spectrum, optional
    error bars, the total fitted model, and individual Gaussian components. A
    lower panel shows the residual ``spectrum - result.best_model`` with a
    shaded one-sigma uncertainty band when ``spectrum_err`` is provided.

    Args:
        velocity: One-dimensional velocity grid.
        spectrum: One-dimensional observed spectrum values.
        spectrum_err: One-dimensional uncertainty values. Pass ``None`` to omit
            error bars and the residual uncertainty band.
        result: Fit result returned by :func:`gaussFitSpec.fit_spectrum`.
        output_file: Optional path where the figure should be saved. Parent
            directories are created automatically.
        show: If ``True``, call :func:`matplotlib.pyplot.show` before returning.
        figure: Optional Matplotlib figure to draw into. If supplied without
            compatible axes, the figure is cleared and rebuilt.
        axes: Optional pair of Matplotlib axes ``(ax_main, ax_residual)``.

    Returns:
        A tuple ``(figure, axes)`` where ``axes`` contains the main spectrum axis
        and the residual axis.

    Example:
        .. code-block:: python

           from gaussFitSpec import plot_fit

           plot_fit(
               velocity,
               spectrum,
               spectrum_err,
               result,
               output_file="examples/example_fit.png",
           )
    """

    velocity = np.asarray(velocity, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    spectrum_err = None if spectrum_err is None else np.asarray(spectrum_err, dtype=float)

    if axes is None or axes[0] is None or len(axes) < 2 or axes[1] is None:
        if figure is None:
            figure, axes = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
                dpi=150,
            )
        else:
            figure.clf()
            axes = figure.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

    ax_main, ax_resid = axes

    if spectrum_err is not None:
        ax_main.errorbar(
            velocity,
            spectrum,
            yerr=spectrum_err,
            fmt="o",
            markersize=2.5,
            linewidth=0.8,
            color="black",
            ecolor="0.7",
            capsize=0,
            label="Spectrum",
        )
    else:
        ax_main.plot(velocity, spectrum, color="black", linewidth=1.0, label="Spectrum")

    ax_main.plot(velocity, result.best_model, color="tab:red", linewidth=1.6, label="Best-fit model")
    for index, component in enumerate(result.individual_components, start=1):
        ax_main.plot(
            velocity,
            component,
            linewidth=1.0,
            alpha=0.85,
            label=f"Component {index}",
        )

    ax_resid.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    if spectrum_err is not None:
        ax_resid.errorbar(
            velocity,
            result.residual,
            yerr=spectrum_err,
            fmt="o",
            markersize=2.5,
            linewidth=0.8,
            color="black",
            ecolor="0.7",
            capsize=0,
        )
        # ax_resid.fill_between(
        #     velocity,
        #     spectrum_err,
        #     -spectrum_err,
        #     color="0.85",
        #     alpha=0.6,
        #     label="1 sigma",
        # )
    else:
        ax_resid.plot(velocity, result.residual, color="black", linewidth=1.0)

    ax_main.set_ylabel("Spectrum")
    ax_resid.set_xlabel("Velocity")
    ax_resid.set_ylabel("Residual")
    ax_main.set_title(f"{result.n_components} Gaussian component(s), method={result.method}")
    ax_main.legend(loc="best", fontsize=8)
    figure.tight_layout()

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes
