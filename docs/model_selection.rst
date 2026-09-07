Model Selection
===============

BIC
---

``method="bic"`` is the default. The package fits models with 1 through
``max_components`` Gaussian components and accepts another component when its
BIC improves by more than ``bic_weight``:

.. math::

   \mathrm{BIC} = \chi^2 + k \ln(n)

where ``k`` is the number of free parameters and ``n`` is the number of data
points.

During BIC model selection, weak components are filtered after the accepted
candidate model is chosen. For manual ``initial_centers`` fits, pass
``filter_components=True`` to apply the same cleanup.

Automatic residual candidates are allowed to overlap an existing fitted center,
so closely spaced Gaussian components can be tested when the data support them.

Absorption Spectra
------------------

By default, Gaussian amplitudes may be positive or negative. For absorption
profiles in ``1 - exp(-tau)`` or tau form, use
``positive_amplitudes=True``. Each component is then constrained during fitting
to an amplitude of at least three times the local uncertainty:

.. code-block:: python

   result = fit_spectrum(
       velocity,
       absorption,
       absorption_err,
       method="bic",
       positive_amplitudes=True,
   )

If manual centers are supplied, ``initial_center_window`` limits how far each
component center can move during optimization. This is useful for a second-pass
fit that should refine, rather than rediscover, the first-pass components.

F-test
------

``method="f_test"`` adds one component at a time and keeps the additional
component only when the approximate F-test reports:

.. code-block:: text

   p < f_test_alpha

The default threshold is ``f_test_alpha=0.05``.

The F-test implementation is an approximate nested-model comparison based on
the improvement in residual sum of squares. It is useful for practical model
selection, but it should not be treated as a fully rigorous astrophysical
line-identification test.
