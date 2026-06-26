from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import make_synthetic_spectrum

from gaussFitSpec import fit_spectrum
from gaussFitSpec.fitting import gaussian


def test_fit_spectrum_cleans_unsorted_nonfinite_and_bad_uncertainties():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(3.0, -1.2, 0.8)])
    velocity = velocity[::-1]
    spectrum = spectrum[::-1]
    spectrum_err = spectrum_err[::-1]
    spectrum[5] = np.nan
    spectrum_err[12] = 0.0

    with pytest.warns(RuntimeWarning, match="Dropping non-finite rows"):
        with pytest.warns(RuntimeWarning, match="Replacing non-positive spectrum_err"):
            result = fit_spectrum(velocity, spectrum, spectrum_err, fixed_n_components=1)

    assert result.best_model.shape == (velocity.size - 1,)
    assert result.parameters == pytest.approx([3.0, -1.2, 0.8], abs=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "not-a-method"}, "method must be either"),
        ({"max_components": 0}, "max_components must be >= 1"),
        ({"fixed_n_components": 0}, "fixed_n_components must be >= 1"),
        ({"initial_centers": []}, "initial_centers must contain at least one center"),
    ],
)
def test_fit_spectrum_rejects_invalid_arguments(kwargs, message):
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(2.0, 0.5, 1.0)])

    with pytest.raises(ValueError, match=message):
        fit_spectrum(velocity, spectrum, spectrum_err, **kwargs)


def test_fit_spectrum_rejects_shape_mismatches():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(2.0, 0.5, 1.0)])

    with pytest.raises(ValueError, match="one-dimensional arrays of equal length"):
        fit_spectrum(velocity[:-1], spectrum, spectrum_err)

    with pytest.raises(ValueError, match="spectrum_err must match the length"):
        fit_spectrum(velocity, spectrum, spectrum_err[:-1])


def test_fit_spectrum_handles_minimum_sample_count():
    velocity = np.array([-1.0, 0.0, 1.0])
    spectrum = gaussian(velocity, 2.0, 0.0, 0.4)
    spectrum_err = np.full_like(velocity, 0.05)

    result = fit_spectrum(velocity, spectrum, spectrum_err, fixed_n_components=1)

    assert result.n_components == 1
    assert np.all(np.isfinite(result.parameters))
    assert result.best_model.shape == velocity.shape


def test_fit_spectrum_rejects_too_few_valid_samples_after_cleaning():
    velocity = np.array([-1.0, 0.0, 1.0])
    spectrum = np.array([1.0, np.nan, 0.5])
    spectrum_err = np.array([0.1, 0.1, 0.1])

    with pytest.warns(RuntimeWarning, match="Dropping non-finite rows"):
        with pytest.raises(ValueError, match="Need at least three valid samples"):
            fit_spectrum(velocity, spectrum, spectrum_err, fixed_n_components=1)


def test_fit_spectrum_replaces_all_non_positive_uncertainties():
    velocity, spectrum, spectrum_err = make_synthetic_spectrum([(2.0, 0.5, 0.8)])
    spectrum_err = np.zeros_like(spectrum_err)

    with pytest.warns(RuntimeWarning, match="Replacing non-positive spectrum_err"):
        result = fit_spectrum(velocity, spectrum, spectrum_err, fixed_n_components=1)

    assert result.n_components == 1
    assert result.fit_statistics["noise_median"] > 0
    assert result.parameters == pytest.approx([2.0, 0.5, 0.8], abs=1e-4)

