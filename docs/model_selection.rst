Model Selection
===============

BIC
---

``method="bic"`` is the default. The package fits models with 1 through
``max_components`` Gaussian components and selects the model with the lowest BIC:

.. math::

   \mathrm{BIC} = \chi^2 + k \ln(n)

where ``k`` is the number of free parameters and ``n`` is the number of data
points.

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
