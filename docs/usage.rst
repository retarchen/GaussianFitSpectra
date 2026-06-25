Basic Usage
===========

Fit a spectrum with the default BIC model selection:

.. code-block:: python

   from gaussFitSpec import fit_spectrum, plot_fit, read_spectrum

   velocity, spectrum, spectrum_err = read_spectrum("examples/example_spectra.txt")

   result = fit_spectrum(
       velocity,
       spectrum,
       spectrum_err,
       name="spectra1",
       method="bic",
       max_components=8,
   )

   print(result.components)
   result.to_csv("examples/example_components.csv")

   plot_fit(
       velocity,
       spectrum,
       spectrum_err,
       result,
   output_file="examples/example_fit.png",
   )

Fitting Options
---------------

Use ``min_fwhm`` and ``max_fwhm`` to constrain component widths:

.. code-block:: python

   result = fit_spectrum(
       velocity,
       spectrum,
       spectrum_err,
       min_fwhm=2.0,
       max_fwhm=40.0,
   )

Fit Result
----------

The returned ``SpectrumFitResult`` contains:

``components``
   A pandas DataFrame with one row per Gaussian component.

``best_model``
   The total fitted Gaussian model evaluated on the input velocity grid.

``individual_components``
   A 2D array containing each Gaussian component evaluated on the input velocity
   grid.

``residual``
   ``spectrum - best_model``.

``method``
   The selected model-selection method.

``n_components``
   The number of selected Gaussian components.

``fit_statistics``
   A dictionary with values such as ``chi2``, ``reduced_chi2``, ``aic``, and
   ``bic``.

Component Table
---------------

The component table includes:

.. code-block:: text

   name
   component_index
   amplitude
   amplitude_err
   velocity
   velocity_err
   fwhm
   fwhm_err
