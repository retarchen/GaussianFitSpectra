from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import make_synthetic_spectrum

from gaussFitSpec import fit_spectrum
from gaussFitSpec.fitting import FWHM_FACTOR


def test_fit_spectrum_recovers_single_component_with_fixed_n(tmp_path):
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(3.0, -1.2, 0.8)])

    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        name="synthetic-single",
        fixed_n_components=1,
    )

    assert result.method == "bic"
    assert result.n_components == 1
    assert result.fit_statistics["n_parameters"] == 3
    assert result.fit_statistics["noise_median"] == pytest.approx(0.05)
    assert result.residual.max() == pytest.approx(0.0)
    assert result.residual.min() == pytest.approx(0.0)
    assert result.best_model.shape == velocity.shape
    assert result.individual_components.shape == (1, velocity.size)
    assert result.parameters == pytest.approx([3.0, -1.2, 0.8], abs=1e-4)

    component = result.components.iloc[0]
    assert component["name"] == "synthetic-single"
    assert component["component_index"] == 1
    assert component["amplitude"] == pytest.approx(3.0, abs=1e-4)
    assert component["velocity"] == pytest.approx(-1.2, abs=1e-4)
    assert component["fwhm"] == pytest.approx(0.8 * FWHM_FACTOR, abs=1e-4)

    csv_path = tmp_path / "single_component.csv"
    result.to_csv(csv_path)
    written = pd.read_csv(csv_path)
    assert list(written.columns) == list(result.components.columns)
    assert written.shape == (1, len(result.components.columns))


def test_bic_selects_two_components_and_sorts_by_velocity():
    components = [(2.5, -3.0, 0.7), (1.8, 2.2, 1.1)]
    velocity, spectrum, spectrum_err = make_synthetic_spectrum(components)

    result = fit_spectrum(velocity, spectrum, spectrum_err, method="bic", max_components=4)

    assert result.n_components == 2
    assert result.fit_statistics["n_parameters"] == 6
    assert result.fit_statistics["bic_history"][0]["n_components"] == 1
    assert result.components["velocity"].to_list() == pytest.approx([-3.0, 2.2], abs=1e-4)
    assert result.parameters == pytest.approx([2.5, -3.0, 0.7, 1.8, 2.2, 1.1], abs=1e-4)


def test_bic_verbose_output_and_weight_are_configurable(capsys):
    components = [(2.5, -3.0, 0.7), (1.8, 2.2, 1.1)]
    velocity, spectrum, spectrum_err = make_synthetic_spectrum(components)

    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        method="bic",
        max_components=4,
        bic_weight=5.0,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "BIC" in captured.out
    assert "n=1" in captured.out
    assert "final BIC=" in captured.out
    assert result.n_components >= 1


def test_initial_centers_override_model_selection_and_sort_output():
    components = [(2.4, -4.0, 0.9), (1.6, 3.5, 0.7)]
    velocity, spectrum, spectrum_err = make_synthetic_spectrum(components)

    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        initial_centers=[3.2, -4.2],
        method="f_test",
        max_components=6,
    )

    assert result.n_components == 2
    assert result.method == "f_test"
    assert result.components["velocity"].to_list() == pytest.approx([-4.0, 3.5], abs=1e-4)


def test_f_test_accepts_second_component_when_improvement_is_significant():
    components = [(3.0, -4.0, 0.7), (2.2, 3.5, 0.9)]
    velocity, spectrum, spectrum_err = make_synthetic_spectrum(components, noise_sigma=0.08, seed=2)

    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        method="f_test",
        max_components=4,
        f_test_alpha=0.05,
    )

    assert result.n_components == 2
    assert result.fit_statistics["f_value"] > 0
    assert result.fit_statistics["f_test_pvalue"] < 0.05
    assert result.parameters == pytest.approx([3.0, -4.0, 0.7, 2.2, 3.5, 0.9], abs=0.06)
