from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import make_synthetic_spectrum

from gaussFitSpec import fit_spectrum


def test_fit_spectrum_handles_flat_spectrum():
    velocity = np.linspace(-5.0, 5.0, 201)
    spectrum = np.zeros_like(velocity)
    spectrum_err = np.full_like(velocity, 0.05)

    result = fit_spectrum(velocity, spectrum, spectrum_err, fixed_n_components=1)

    assert result.n_components == 1
    assert abs(result.components.loc[0, "amplitude"]) < 1e-6
    assert np.max(np.abs(result.best_model)) < 1e-6


def test_fit_spectrum_recovers_negative_amplitude_component():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(-2.0, 0.8, 0.7)])

    result = fit_spectrum(velocity, spectrum, spectrum_err, fixed_n_components=1)

    assert result.n_components == 1
    assert result.parameters == pytest.approx([-2.0, 0.8, 0.7], abs=1e-4)


def test_fit_spectrum_enforces_max_fwhm_bound():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(2.5, 0.0, 1.5)])

    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        fixed_n_components=1,
        max_fwhm=1.0,
    )

    assert result.n_components == 1
    assert result.components.loc[0, "fwhm"] == pytest.approx(1.0, abs=1e-4)


def test_fit_spectrum_enforces_min_fwhm_bound():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(2.5, 0.0, 0.15)])

    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        fixed_n_components=1,
        min_fwhm=1.0,
    )

    assert result.n_components == 1
    assert result.components.loc[0, "fwhm"] == pytest.approx(1.0, abs=1e-4)


def test_bic_prefers_one_component_for_strongly_overlapping_lines():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum(
        [(1.5, 0.0, 0.7), (1.0, 0.25, 0.7)]
    )

    result = fit_spectrum(velocity, spectrum, spectrum_err, method="bic", max_components=4)

    assert result.n_components == 1
    assert result.fit_statistics["n_parameters"] == 3

