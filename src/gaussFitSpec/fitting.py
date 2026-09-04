"""Core multi-Gaussian fitting routines."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class SpectrumFitResult:
    """Result returned by :func:`fit_spectrum`.

    The object stores both the human-readable component table and the arrays
    needed for downstream analysis or plotting.

    Attributes:
        components: Table with one row per fitted Gaussian component. The
            columns are ``name``, ``component_index``, ``amplitude``,
            ``amplitude_err``, ``velocity``, ``velocity_err``, ``fwhm``, and
            ``fwhm_err``.
        best_model: Total fitted model evaluated at the input velocity values.
        individual_components: Two-dimensional array with shape
            ``(n_components, n_velocity)`` containing each Gaussian component.
        residual: Difference between the input spectrum and ``best_model``.
        method: Model-selection method used for the fit, either ``"bic"`` or
            ``"f_test"``.
        n_components: Number of Gaussian components selected.
        fit_statistics: Dictionary containing goodness-of-fit values such as
            ``chi2``, ``reduced_chi2``, ``aic``, ``bic``, ``rss``, and
            ``n_parameters``. F-test fits also include ``f_value`` and
            ``f_test_pvalue`` when applicable.
        covariance: Covariance matrix returned by ``scipy.optimize.curve_fit``.
            This is ``None`` only if no usable covariance is available.
        parameters: Raw fitted parameters in repeating
            ``amplitude, velocity, sigma`` order.
    """

    components: pd.DataFrame
    best_model: np.ndarray
    individual_components: np.ndarray
    residual: np.ndarray
    method: str
    n_components: int
    fit_statistics: dict
    covariance: np.ndarray | None = None
    parameters: np.ndarray | None = None

    def to_csv(self, path, index=False):
        """Write the component table to a CSV file.

        Args:
            path: Output CSV file path.
            index: Passed to :meth:`pandas.DataFrame.to_csv`. The default
                ``False`` avoids writing the DataFrame index.
        """

        self.components.to_csv(path, index=index)


def gaussian(x, amplitude, center, sigma):
    """Evaluate a single Gaussian component.

    Args:
        x: Positions where the Gaussian should be evaluated.
        amplitude: Peak amplitude. Positive and negative amplitudes are both
            allowed.
        center: Gaussian center in the same units as ``x``.
        sigma: Gaussian standard deviation. Use
            ``FWHM = 2 * sqrt(2 * ln(2)) * sigma`` to convert to FWHM.

    Returns:
        A NumPy array containing the Gaussian values at ``x``.
    """

    x = np.asarray(x, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def multi_gaussian(x, *params):
    """Evaluate the sum of one or more Gaussian components.

    Args:
        x: Positions where the model should be evaluated.
        *params: Flat parameter list in repeating
            ``amplitude, center, sigma`` order. For example, two components are
            represented as ``amp1, center1, sigma1, amp2, center2, sigma2``.

    Returns:
        A NumPy array containing the summed Gaussian model.

    Raises:
        ValueError: The underlying numerical operations may fail if a supplied
            ``sigma`` is zero or non-finite.
    """

    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x, dtype=float)
    for index in range(0, len(params), 3):
        total += gaussian(x, *params[index : index + 3])
    return total


def fit_spectrum(
    velocity,
    spectrum,
    spectrum_err=None,
    *,
    name="spectra1",
    method="bic",
    max_components=8,
    f_test_alpha=0.05,
    min_fwhm=None,
    max_fwhm=None,
    initial_centers=None,
    initial_center_window=None,
    fixed_n_components=None,
    bic_weight=10.0,
    positive_amplitudes=False,
    filter_components=False,
    verbose=False,
):
    """Fit a 1D spectrum with multiple Gaussian components.

    The fitter builds candidate models sequentially, using peaks in the
    spectrum or residuals as initial guesses. Gaussian amplitudes may be
    positive or negative by default, component centers are bounded by the input
    velocity range, and widths are constrained by ``min_fwhm`` and ``max_fwhm``
    when provided.

    Args:
        velocity: One-dimensional velocity grid.
        spectrum: One-dimensional spectrum values sampled on ``velocity``.
        spectrum_err: One-dimensional uncertainty values. If ``None``, a single
            noise level is estimated from the spectrum and used for all points.
        name: Name stored in the output component table. The default is
            ``"spectra1"``.
        method: Model-selection method. Use ``"bic"`` for Bayesian information
            criterion selection or ``"f_test"`` for the approximate sequential
            F-test.
        max_components: Maximum number of Gaussian components to try when
            selecting a model automatically.
        f_test_alpha: Significance threshold used by ``method="f_test"``.
            Additional components are accepted when the approximate F-test
            p-value is below this value.
        min_fwhm: Optional minimum allowed full width at half maximum.
        max_fwhm: Optional maximum allowed full width at half maximum.
        initial_centers: Optional iterable of component centers. When provided,
            the fitter uses these centers as the initial model instead of
            automatic model selection.
        initial_center_window: Optional maximum absolute distance each fitted
            center may move from its corresponding ``initial_centers`` value.
            This is only used when ``initial_centers`` is provided.
        fixed_n_components: Optional fixed number of components to fit. When
            provided, automatic model selection is skipped.
        bic_weight: Minimum BIC improvement required before accepting an
            additional Gaussian component when ``method="bic"``. This also
            controls the extra penalty term applied to weak components, matching
            the behavior of the reference fitting script.
        positive_amplitudes: If ``True``, constrain Gaussian amplitudes to be
            non-negative. Use this for absorption spectra represented as
            ``1 - exp(-tau)`` or tau, where negative components are unphysical.
        filter_components: If ``True``, apply the weak-component filter after a
            manual ``initial_centers`` fit, matching the automatic BIC cleanup.
        verbose: If ``True``, print per-iteration progress messages during
            sequential model selection. For BIC fits this includes the BIC
            value and number of Gaussian components tried.

    Returns:
        A :class:`SpectrumFitResult` containing the component table, model
        arrays, residuals, selected method, number of components, covariance,
        and fit statistics.

    Raises:
        ValueError: If input arrays have incompatible shapes, if ``method`` is
            not ``"bic"`` or ``"f_test"``, or if ``max_components`` is less than
            one.
        RuntimeError: If no candidate Gaussian model can be fitted.

    Example:
        .. code-block:: python

           from gaussFitSpec import fit_spectrum, read_spectrum

           velocity, spectrum, spectrum_err = read_spectrum("examples/example_spectra.txt")
           result = fit_spectrum(velocity, spectrum, spectrum_err, method="bic")
           print(result.components)
    """

    velocity, spectrum, spectrum_err = _prepare_inputs(velocity, spectrum, spectrum_err)
    method_normalized = method.lower()
    if method_normalized not in {"bic", "f_test"}:
        raise ValueError("method must be either 'bic' or 'f_test'.")
    if max_components < 1:
        raise ValueError("max_components must be >= 1.")

    fitter = _SequentialGaussianFitter(
        velocity=velocity,
        spectrum=spectrum,
        spectrum_err=spectrum_err,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        bic_weight=float(bic_weight),
        positive_amplitudes=bool(positive_amplitudes),
        verbose=bool(verbose),
    )

    if initial_centers is not None:
        params, covariance, stats_dict = fitter.fit_with_centers(
            np.asarray(list(initial_centers), dtype=float),
            center_window=initial_center_window,
            filter_components=bool(filter_components),
        )
    elif fixed_n_components is not None:
        params, covariance, stats_dict = fitter.fit_fixed_n(int(fixed_n_components))
    elif method_normalized == "bic":
        params, covariance, stats_dict = fitter.select_bic(max_components=int(max_components))
    else:
        params, covariance, stats_dict = fitter.select_f_test(
            max_components=int(max_components),
            alpha=float(f_test_alpha),
        )

    return _build_result(
        velocity=velocity,
        spectrum=spectrum,
        spectrum_err=spectrum_err,
        name=name,
        method=method_normalized,
        params=params,
        covariance=covariance,
        fit_statistics=stats_dict,
    )


def _prepare_inputs(velocity, spectrum, spectrum_err):
    velocity = np.asarray(velocity, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if velocity.ndim != 1 or spectrum.ndim != 1 or velocity.size != spectrum.size:
        raise ValueError("velocity and spectrum must be one-dimensional arrays of equal length.")

    if spectrum_err is None:
        estimated = _estimate_noise(spectrum)
        spectrum_err = np.full_like(spectrum, estimated, dtype=float)
    else:
        spectrum_err = np.asarray(spectrum_err, dtype=float)
        if spectrum_err.ndim != 1 or spectrum_err.size != spectrum.size:
            raise ValueError("spectrum_err must match the length of velocity and spectrum.")

    finite_mask = np.isfinite(velocity) & np.isfinite(spectrum) & np.isfinite(spectrum_err)
    if not np.all(finite_mask):
        warnings.warn(
            "Dropping non-finite rows before fitting.",
            RuntimeWarning,
            stacklevel=2,
        )
    velocity = velocity[finite_mask]
    spectrum = spectrum[finite_mask]
    spectrum_err = spectrum_err[finite_mask]
    if velocity.size < 3:
        raise ValueError("Need at least three valid samples to fit a spectrum.")

    positive_mask = spectrum_err > 0
    if not np.all(positive_mask):
        replacement = _estimate_noise(spectrum)
        warnings.warn(
            "Replacing non-positive spectrum_err values with an estimated noise level.",
            RuntimeWarning,
            stacklevel=2,
        )
        spectrum_err = spectrum_err.copy()
        spectrum_err[~positive_mask] = replacement

    order = np.argsort(velocity)
    return velocity[order], spectrum[order], spectrum_err[order]


def _estimate_noise(spectrum):
    if spectrum.size < 2:
        return 1.0
    diff = np.diff(spectrum)
    mad = np.median(np.abs(diff - np.median(diff)))
    sigma = mad / 0.67448975 / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(spectrum)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return float(sigma)


class _SequentialGaussianFitter:
    """Sequential model builder inspired by the original script logic."""

    def __init__(
        self,
        velocity,
        spectrum,
        spectrum_err,
        min_fwhm=None,
        max_fwhm=None,
        bic_weight=10.0,
        positive_amplitudes=False,
        verbose=False,
    ):
        self.velocity = velocity
        self.spectrum = spectrum
        self.spectrum_err = spectrum_err
        self.bic_weight = float(bic_weight)
        self.positive_amplitudes = bool(positive_amplitudes)
        self.verbose = bool(verbose)
        self.noise = float(np.nanmedian(np.abs(spectrum_err)))
        self.dx = float(np.nanmedian(np.diff(velocity)))
        velocity_span = float(velocity.max() - velocity.min())
        self.min_sigma = max(
            (abs(self.dx) if np.isfinite(self.dx) and self.dx != 0 else velocity_span / max(len(velocity) - 1, 1))
            / FWHM_FACTOR,
            (min_fwhm or max(abs(self.dx) * 2.0, velocity_span / 200.0)) / FWHM_FACTOR,
        )
        default_max_fwhm = max(velocity_span / 2.0, self.min_sigma * FWHM_FACTOR * 2.0)
        self.max_sigma = max((max_fwhm or default_max_fwhm) / FWHM_FACTOR, self.min_sigma * 1.5)
        self.amp_bound = max(5.0 * self.noise, 2.5 * np.max(np.abs(self.spectrum)))

    def fit_with_centers(self, centers, center_window=None, filter_components=False):
        if centers.size == 0:
            raise ValueError("initial_centers must contain at least one center.")
        params = self._initial_params_from_centers(centers, self.spectrum)
        params, covariance, stats_dict = self._fit_model(params, center_window=center_window)
        if filter_components:
            bic_history = [{"n_components": len(params) // 3, "bic": float(stats_dict["bic"])}]
            params, covariance, stats_dict = self._refit_filtered_components(
                params,
                covariance,
                stats_dict,
                bic_history,
                center_window=center_window,
            )
        return params, covariance, stats_dict

    def fit_fixed_n(self, n_components):
        if n_components < 1:
            raise ValueError("fixed_n_components must be >= 1.")
        params = None
        covariance = None
        stats_dict = None
        for n_current in range(1, n_components + 1):
            params, covariance, stats_dict = self._fit_next_model(n_current, params)
        return params, covariance, stats_dict

    def select_bic(self, max_components):
        centers = self._initial_bic_candidate_centers()
        if centers.size == 0:
            centers = np.array([self.velocity[int(np.argmax(self.spectrum))]], dtype=float)

        params0 = self._initial_params_from_centers(centers[:1], self.spectrum)
        params, covariance, stats_dict = self._fit_model(params0)
        bic_history = [{"n_components": 1, "bic": float(stats_dict["bic"])}]
        if self.verbose:
            print(f"BIC {stats_dict['bic']:.6f} n=1")

        best_params = params
        best_covariance = covariance
        best_stats = dict(stats_dict)
        best_stats["bic_history"] = list(bic_history)
        current_params = params
        current_centers = list(np.asarray(current_params)[1::3])
        residual = self.spectrum - multi_gaussian(self.velocity, *current_params)
        smoothed_residual = self._smooth_residual(residual)

        while len(current_centers) < max_components:
            candidate_centers = self._candidate_centers_from_signal(smoothed_residual, used_centers=current_centers)
            if candidate_centers.size == 0:
                break

            best_candidate = None
            for new_center in candidate_centers:
                new_guess = self._initial_params_from_centers(np.array([new_center]), residual)
                trial_params0 = np.concatenate([current_params, new_guess])
                try:
                    trial_params, trial_covariance, trial_stats = self._fit_model(trial_params0)
                except RuntimeError:
                    continue
                if best_candidate is None or trial_stats["bic"] < best_candidate[2]["bic"]:
                    best_candidate = (trial_params, trial_covariance, trial_stats)

            if best_candidate is None:
                break

            current_params, current_covariance, current_stats = best_candidate
            n_components = len(current_params) // 3
            bic_history.append({"n_components": n_components, "bic": float(current_stats["bic"])})
            if self.verbose:
                print(f"BIC {current_stats['bic']:.6f} n={n_components}")

            improvement = best_stats["bic"] - current_stats["bic"]
            if improvement > self.bic_weight:
                best_params = current_params
                best_covariance = current_covariance
                best_stats = dict(current_stats)
                best_stats["bic_history"] = list(bic_history)
            else:
                break

            current_centers = list(np.asarray(current_params)[1::3])
            residual = self.spectrum - multi_gaussian(self.velocity, *current_params)
            smoothed_residual = self._smooth_residual(residual)

        best_params, best_covariance, best_stats = self._refit_filtered_components(
            best_params,
            best_covariance,
            best_stats,
            bic_history,
        )

        if best_params is None:
            raise RuntimeError("No valid fits were found.")
        if self.verbose:
            print(f"final BIC={best_stats['bic']:.6f} n={len(best_params) // 3}")
        return best_params, best_covariance, best_stats

    def select_f_test(self, max_components, alpha):
        accepted = None
        previous_params = None
        previous_stats = None
        previous_covariance = None
        for n_components in range(1, max_components + 1):
            params, covariance, stats_dict = self._fit_next_model(n_components, previous_params)
            if accepted is None:
                accepted = (params, covariance, stats_dict)
                previous_params = params
                previous_covariance = covariance
                previous_stats = stats_dict
                continue

            dof_diff = len(params) - len(previous_params)
            denom_dof = len(self.velocity) - len(params)
            if dof_diff <= 0 or denom_dof <= 0 or previous_stats["rss"] <= stats_dict["rss"]:
                break

            numerator = (previous_stats["rss"] - stats_dict["rss"]) / dof_diff
            denominator = stats_dict["rss"] / denom_dof
            if denominator <= 0:
                break
            f_value = numerator / denominator
            p_value = stats.f.sf(f_value, dof_diff, denom_dof)
            stats_dict["f_value"] = float(f_value)
            stats_dict["f_test_pvalue"] = float(p_value)
            if np.isfinite(p_value) and p_value < alpha:
                accepted = (params, covariance, stats_dict)
                previous_params = params
                previous_covariance = covariance
                previous_stats = stats_dict
            else:
                break

        if accepted is None:
            raise RuntimeError("No valid fits were found.")
        if "f_test_pvalue" not in accepted[2]:
            accepted[2]["f_test_pvalue"] = np.nan
            accepted[2]["f_value"] = np.nan
        return accepted

    def _fit_next_model(self, n_components, previous_params):
        if n_components == 1 or previous_params is None:
            centers = self._seed_centers(1)
            params0 = self._initial_params_from_centers(centers, self.spectrum)
        else:
            centers = self._seed_centers(n_components, previous_params=previous_params)
            new_center = centers[-1]
            residual = self.spectrum - multi_gaussian(self.velocity, *previous_params)
            new_guess = self._initial_params_from_centers(np.array([new_center]), residual)
            params0 = np.concatenate([previous_params, new_guess])
        return self._fit_model(params0)

    def _fit_model(self, params0, center_window=None):
        center_guesses = None
        if center_window is not None:
            center_guesses = np.asarray(params0, dtype=float)[1::3]
        lower_bounds, upper_bounds = self._parameter_bounds(
            len(params0) // 3,
            center_guesses=center_guesses,
            center_window=center_window,
        )
        params0 = np.clip(params0, lower_bounds + 1e-6, upper_bounds - 1e-6)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, covariance = curve_fit(
                    multi_gaussian,
                    self.velocity,
                    self.spectrum,
                    p0=params0,
                    sigma=self.spectrum_err,
                    absolute_sigma=True,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=200000,
                )
        except Exception as exc:
            raise RuntimeError(f"Gaussian fit failed: {exc}") from exc

        params, covariance = self._sort_fit(params, covariance)
        stats_dict = self._fit_statistics(params)
        return params, covariance, stats_dict

    def _fit_statistics(self, params):
        model = multi_gaussian(self.velocity, *params)
        residual = self.spectrum - model
        chi2 = float(np.sum((residual / self.spectrum_err) ** 2))
        rss = float(np.sum(residual**2))
        n_samples = len(self.velocity)
        n_params = len(params)
        dof = max(n_samples - n_params, 1)
        bic = chi2 + n_params * np.log(n_samples)
        weak_component_floor = max(
            abs(float(np.min(self.spectrum))),
            float(np.max(np.abs(params[0::3]))) / 5.0 if len(params) else 0.0,
            _estimate_component_noise(self.spectrum, self.spectrum_err, n=10) * 4.0,
        )
        weak_component_count = int(np.sum(np.abs(params[0::3]) < weak_component_floor))
        bic += self.bic_weight * weak_component_count
        return {
            "chi2": chi2,
            "reduced_chi2": chi2 / dof,
            "aic": chi2 + 2.0 * n_params,
            "bic": float(bic),
            "rss": rss,
            "n_parameters": n_params,
        }

    def _seed_centers(self, n_components, previous_params=None):
        if previous_params is not None:
            used = list(np.asarray(previous_params)[1::3])
            residual = self.spectrum - multi_gaussian(self.velocity, *previous_params)
        else:
            used = []
            residual = self.spectrum

        centers = list(used)
        while len(centers) < n_components:
            centers.append(self._pick_center(residual, used_centers=centers))
        return np.asarray(centers, dtype=float)

    def _pick_center(self, residual, used_centers):
        smoothed = gaussian_filter1d(residual, sigma=1.0, mode="nearest")
        prominence = max(self.noise, np.std(smoothed) / 5.0)
        min_distance = max(1, int(np.ceil(2.0 * self.min_sigma / max(abs(self.dx), 1e-6))))

        peak_indices, _ = find_peaks(smoothed, prominence=prominence, distance=min_distance)
        dip_indices, _ = find_peaks(-smoothed, prominence=prominence, distance=min_distance)
        candidates = np.unique(np.concatenate([peak_indices, dip_indices, [int(np.argmax(np.abs(smoothed)))]]))
        candidates = sorted(candidates, key=lambda idx: abs(smoothed[idx]), reverse=True)

        for idx in candidates:
            center = float(self.velocity[idx])
            if all(abs(center - existing) >= max(self.min_sigma * FWHM_FACTOR, abs(self.dx) * 2.0) for existing in used_centers):
                return center

        return float(self.velocity[int(np.argmax(np.abs(residual)))])

    def _initial_params_from_centers(self, centers, reference_signal):
        params = []
        default_sigma = np.clip(abs(self.dx), self.min_sigma, self.max_sigma)
        for center in centers:
            idx = int(np.argmin(np.abs(self.velocity - center)))
            start = max(idx - 5, 0)
            stop = min(idx + 6, self.spectrum_err.size)
            local_noise = float(np.nanmean(np.abs(self.spectrum_err[start:stop])))
            amplitude = local_noise
            if self.positive_amplitudes:
                amplitude = max(float(amplitude), self.noise)
            if not np.isfinite(amplitude) or abs(amplitude) < self.noise:
                amplitude = np.sign(amplitude) * self.noise if amplitude != 0 else self.noise
            params.extend([float(amplitude), float(self.velocity[idx]), float(default_sigma)])
        return np.asarray(params, dtype=float)

    def _initial_bic_candidate_centers(self):
        peak_indices, _ = find_peaks(
            self.spectrum,
            height=np.max(self.spectrum) / 5.0 if np.max(self.spectrum) > 0 else self.noise,
            distance=5,
        )
        if peak_indices.size == 0:
            return np.array([self.velocity[int(np.argmax(self.spectrum))]], dtype=float)

        order = np.argsort(self.spectrum[peak_indices])[::-1]
        ordered = peak_indices[order]
        edge_buffer = max(self.velocity_span / 20.0, abs(self.dx) * 10.0)
        mask = (self.velocity[ordered] > self.velocity.min() + edge_buffer) & (
            self.velocity[ordered] < self.velocity.max() - edge_buffer
        )
        filtered = ordered[mask]
        if filtered.size == 0:
            filtered = ordered
        return self.velocity[filtered]

    @property
    def velocity_span(self):
        return float(self.velocity.max() - self.velocity.min())

    def _smooth_residual(self, residual):
        kernel = np.array([0.054489, 0.244201, 0.40262, 0.244201, 0.054489], dtype=float)
        return convolve1d(residual, kernel, mode="nearest")

    def _candidate_centers_from_signal(self, signal, used_centers):
        height = np.max(signal) / 5.0 if np.max(signal) > 0 else self.noise
        peak_indices, _ = find_peaks(signal, height=height, distance=5)
        if peak_indices.size == 0:
            peak_indices = np.array([int(np.argmax(signal))], dtype=int)
        order = np.argsort(signal[peak_indices])[::-1]
        ordered = peak_indices[order]
        edge_buffer = max(self.velocity_span / 20.0, abs(self.dx) * 10.0)
        ordered = ordered[
            (self.velocity[ordered] > self.velocity.min() + edge_buffer)
            & (self.velocity[ordered] < self.velocity.max() - edge_buffer)
        ]
        if ordered.size == 0:
            return np.array([], dtype=float)

        candidates = []
        min_separation = max(self.min_sigma * FWHM_FACTOR, abs(self.dx) * 2.0)
        for idx in ordered[:5]:
            center = float(self.velocity[idx])
            if all(abs(center - existing) >= min_separation for existing in used_centers):
                candidates.append(center)
        return np.asarray(candidates, dtype=float)

    def _refit_filtered_components(self, params, covariance, stats_dict, bic_history, center_window=None):
        if params is None or len(params) <= 3:
            stats_dict = dict(stats_dict)
            stats_dict["bic_history"] = list(bic_history)
            return params, covariance, stats_dict

        component_rows = params.reshape(-1, 3)
        filtered_rows = []
        for row in component_rows:
            noise_here = _estimate_component_noise_at_center(row[1], self.velocity, self.spectrum_err)
            if row[0] > noise_here * 3.0:
                filtered_rows.append(row)

        if not filtered_rows:
            filtered_rows = [component_rows[int(np.argmax(component_rows[:, 0]))]]

        filtered_params = np.asarray(filtered_rows, dtype=float).reshape(-1)
        final_floor = max(
            _estimate_component_noise(self.spectrum, self.spectrum_err, n=10) * 3.0,
            abs(float(np.min(self.spectrum))) * 0.8,
        )
        filtered_matrix = filtered_params.reshape(-1, 3)
        strongest = int(np.argmax(filtered_matrix[:, 0]))
        keep_mask = ~(
            (filtered_matrix[:, 0] < final_floor)
            & (np.abs(filtered_matrix[:, 1] - filtered_matrix[strongest, 1]) > 20.0)
        )
        filtered_matrix = filtered_matrix[keep_mask]
        filtered_params = filtered_matrix.reshape(-1)

        final_params, final_covariance, final_stats = self._fit_model(
            filtered_params,
            center_window=center_window,
        )
        final_stats["bic_history"] = list(bic_history)
        return final_params, final_covariance, final_stats

    def _parameter_bounds(self, n_components, center_guesses=None, center_window=None):
        lower_bounds = []
        upper_bounds = []
        amp_lower = 0.0 if self.positive_amplitudes else -self.amp_bound
        if center_window is not None:
            center_window = float(center_window)
            if center_window <= 0:
                raise ValueError("initial_center_window must be positive.")
            center_guesses = np.asarray(center_guesses, dtype=float)
            if center_guesses.size != n_components:
                raise ValueError("center_guesses must match the number of components.")
        for index in range(n_components):
            if center_window is None:
                center_low = float(self.velocity.min())
                center_high = float(self.velocity.max())
            else:
                center_low = max(float(self.velocity.min()), float(center_guesses[index]) - center_window)
                center_high = min(float(self.velocity.max()), float(center_guesses[index]) + center_window)
            lower_bounds.extend([amp_lower, center_low, self.min_sigma])
            upper_bounds.extend([self.amp_bound, center_high, self.max_sigma])
        return np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float)

    def _sort_fit(self, params, covariance):
        order = np.argsort(params[1::3])
        permutation = []
        sorted_params = np.empty_like(params)
        for new_index, old_index in enumerate(order):
            old_slice = slice(old_index * 3, old_index * 3 + 3)
            new_slice = slice(new_index * 3, new_index * 3 + 3)
            sorted_params[new_slice] = params[old_slice]
            permutation.extend(range(old_index * 3, old_index * 3 + 3))

        if covariance is None or np.ndim(covariance) != 2:
            return sorted_params, covariance
        sorted_covariance = covariance[np.ix_(permutation, permutation)]
        return sorted_params, sorted_covariance


def _build_result(velocity, spectrum, spectrum_err, name, method, params, covariance, fit_statistics):
    best_model = multi_gaussian(velocity, *params)
    residual = spectrum - best_model
    individual_components = np.array(
        [gaussian(velocity, *params[index : index + 3]) for index in range(0, len(params), 3)],
        dtype=float,
    )
    components = _components_dataframe(
        name=name,
        params=params,
        covariance=covariance,
    )
    fit_statistics = dict(fit_statistics)
    fit_statistics["noise_median"] = float(np.median(spectrum_err))
    return SpectrumFitResult(
        components=components,
        best_model=best_model,
        individual_components=individual_components,
        residual=residual,
        method=method,
        n_components=individual_components.shape[0],
        fit_statistics=fit_statistics,
        covariance=covariance,
        parameters=np.asarray(params, dtype=float),
    )


def _components_dataframe(name, params, covariance):
    errors = np.full_like(params, np.nan, dtype=float)
    if covariance is not None and np.ndim(covariance) == 2 and covariance.shape[0] == covariance.shape[1]:
        diagonal = np.diag(covariance)
        valid = np.isfinite(diagonal) & (diagonal >= 0)
        errors[valid] = np.sqrt(diagonal[valid])

    rows = []
    for component_index, start in enumerate(range(0, len(params), 3), start=1):
        amplitude, center, sigma = params[start : start + 3]
        amplitude_err, center_err, sigma_err = errors[start : start + 3]
        fwhm = sigma * FWHM_FACTOR
        fwhm_err = sigma_err * FWHM_FACTOR if np.isfinite(sigma_err) else np.nan
        rows.append(
            {
                "name": name,
                "component_index": component_index,
                "amplitude": float(amplitude),
                "amplitude_err": float(amplitude_err),
                "velocity": float(center),
                "velocity_err": float(center_err),
                "fwhm": float(fwhm),
                "fwhm_err": float(fwhm_err),
            }
        )

    return pd.DataFrame(rows)


def _estimate_component_noise(spectrum, spectrum_err, n=1):
    spectrum = np.asarray(spectrum, dtype=float)
    spectrum_err = np.asarray(spectrum_err, dtype=float)
    positive = n * spectrum_err - spectrum
    indices = np.argwhere(positive > 0).ravel()
    if indices.size == 0:
        return float(np.nanmean(spectrum_err))
    return float(np.nanmean(spectrum_err[indices]))


def _estimate_component_noise_at_center(center, velocity, spectrum_err):
    velocity = np.asarray(velocity, dtype=float)
    spectrum_err = np.asarray(spectrum_err, dtype=float)
    indices = np.argwhere(np.abs(velocity - center) < 5.0).ravel()
    if indices.size == 0:
        return float(np.nanmean(spectrum_err))
    return float(np.nanmean(spectrum_err[indices]))
