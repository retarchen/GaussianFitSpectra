Input Format
============

Input spectrum files must contain exactly three numerical columns in this order:

1. ``velocity``
2. ``spectrum``
3. ``spectrum_err``

Plain text tables such as ``.txt``, ``.dat``, and ``.csv`` are supported when
they follow this three-column structure. Whitespace-separated and comma-separated
tables are both accepted.

Example
-------

.. code-block:: text

   223.526564  0.040043  0.172759
   224.113598  0.180919  0.789550
   224.700631  0.293556  1.416493

Read a file with:

.. code-block:: python

   from gaussFitSpec import read_spectrum

   velocity, spectrum, spectrum_err = read_spectrum("examples/example_spectra.txt")
