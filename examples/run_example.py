"""Example script for fitting the bundled example spectrum."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gaussFitSpec import fit_spectrum, plot_fit, read_spectrum


def main():
    examples_dir = PROJECT_ROOT / "examples"
    input_file = examples_dir / "example_spectra.txt"
    output_csv = examples_dir / "example_components.csv"
    output_png = examples_dir / "example_fit.png"

    velocity, spectrum, spectrum_err = read_spectrum(input_file)
    result = fit_spectrum(
        velocity,
        spectrum,
        spectrum_err,
        name="spectra1",
        method="bic",
        max_components=8,
    )

    result.to_csv(output_csv)
    plot_fit(
        velocity,
        spectrum,
        spectrum_err,
        result,
        output_file=output_png,
    )

    print(result.components)
    print(f"Saved CSV to {output_csv}")
    print(f"Saved figure to {output_png}")


if __name__ == "__main__":
    main()
