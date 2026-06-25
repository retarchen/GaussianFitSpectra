from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussFitSpec import fit_spectrum, plot_fit, read_spectrum, save_components_csv


def test_read_spectrum():
    velocity, spectrum, spectrum_err = read_spectrum(ROOT / "examples" / "example_spectra.txt")
    assert velocity.ndim == 1
    assert velocity.size == spectrum.size == spectrum_err.size
    assert velocity.size > 10


def test_fit_and_outputs(tmp_path):
    velocity, spectrum, spectrum_err = read_spectrum(ROOT / "examples" / "example_spectra.txt")
    result = fit_spectrum(velocity, spectrum, spectrum_err, method="bic", max_components=6)

    assert result.n_components >= 1
    assert result.best_model.shape == spectrum.shape
    assert result.residual.shape == spectrum.shape
    assert set(result.components.columns) == {
        "name",
        "component_index",
        "amplitude",
        "amplitude_err",
        "velocity",
        "velocity_err",
        "fwhm",
        "fwhm_err",
    }

    csv_path = tmp_path / "components.csv"
    save_components_csv(result, csv_path)
    assert csv_path.exists()
    assert csv_path.read_text().strip()

    plot_path = tmp_path / "fit.png"
    figure, _ = plot_fit(velocity, spectrum, spectrum_err, result, output_file=plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    figure.clf()
