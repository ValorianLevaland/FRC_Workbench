import numpy as np
from frc_workbench.core.frc_backend import compute_frc_curve, find_cutoff_frequency_robust


def test_frc_curve_smoke():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 64)).astype(np.float32)
    b = a + 0.05 * rng.normal(size=(64, 64)).astype(np.float32)

    freqs, frc = compute_frc_curve(a, b)
    assert freqs.ndim == 1 and frc.ndim == 1
    assert freqs.size == frc.size
    c = find_cutoff_frequency_robust(freqs, frc, threshold=1/7, smooth_bins=5, nyquist_guard=0.49)
    # cutoff may be None depending on noise; just ensure it doesn't crash and is in range if present
    if c is not None:
        assert 0.0 < c < 0.5
