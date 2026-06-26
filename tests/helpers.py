from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussFitSpec.fitting import gaussian


def make_synthetic_spectrum(components, *, noise_sigma=0.0, seed=0):
    velocity = np.linspace(-15.0, 15.0, 601)
    spectrum = np.zeros_like(velocity)
    for amplitude, center, sigma in components:
        spectrum += gaussian(velocity, amplitude, center, sigma)

    spectrum_err = np.full_like(velocity, max(noise_sigma, 0.05))
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        spectrum = spectrum + rng.normal(0.0, noise_sigma, size=velocity.size)

    return velocity, spectrum, spectrum_err

