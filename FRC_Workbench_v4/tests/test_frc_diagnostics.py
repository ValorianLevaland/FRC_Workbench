import numpy as np
import pandas as pd

from frc_workbench.core import frc_backend
from frc_workbench.core.diagnostics import (
    BACKEND_ROI_APOD_SIGMA_PX,
    BACKEND_ROI_APOD_SIGMA_SOURCE,
    gaussian_sigma_px,
    frc_parameter_metadata,
)
from frc_workbench.core.frc_analysis import _apodize_mask, compute_frc_plot_data
from frc_workbench.core.frc_backend import (
    compute_frc_curve,
    find_cutoff_frequency_robust,
    render_two_images,
)


def _tiny_render_df(offset_nm=0.0):
    return pd.DataFrame(
        {
            "x_nm": np.array([0, 10, 20, 30, 40], dtype=float) + offset_nm,
            "y_nm": np.array([0, 0, 10, 20, 30], dtype=float),
            "frame": np.arange(5),
            "intensity": np.ones(5),
        }
    )


def test_gaussian_sigma_nm_to_pixel_conversion():
    assert gaussian_sigma_px(gaussian_sigma_nm=10.0, pixel_size_nm=5.0) == 2.0

    img_a, img_b, meta = render_two_images(
        _tiny_render_df(),
        _tiny_render_df(offset_nm=5.0),
        pixel_size_nm=5.0,
        gaussian_sigma_nm=10.0,
    )

    assert img_a.shape == img_b.shape
    assert meta["gaussian_sigma_nm"] == 10.0
    assert meta["gaussian_sigma_px"] == 2.0


def test_identical_gaussian_blur_sigma_applied_to_both_halves(monkeypatch):
    calls = []

    def recording_blur(img, sigma_pix):
        calls.append(float(sigma_pix))
        return np.asarray(img, dtype=float)

    monkeypatch.setattr(frc_backend, "gaussian_blur", recording_blur)

    render_two_images(
        _tiny_render_df(),
        _tiny_render_df(offset_nm=5.0),
        pixel_size_nm=5.0,
        gaussian_sigma_nm=7.5,
    )

    assert calls == [1.5, 1.5]


def test_gaussian_sigma_changes_rendered_pixels_but_cutoff_can_remain_stable():
    rng = np.random.default_rng(123)
    n = 500
    base_x = rng.uniform(0, 500, n)
    base_y = rng.uniform(0, 500, n)
    jitter = 3.0
    df_a = pd.DataFrame(
        {
            "x_nm": base_x + rng.normal(0, jitter, n),
            "y_nm": base_y + rng.normal(0, jitter, n),
            "frame": np.arange(n),
            "intensity": np.ones(n),
        }
    )
    df_b = pd.DataFrame(
        {
            "x_nm": base_x + rng.normal(0, jitter, n),
            "y_nm": base_y + rng.normal(0, jitter, n),
            "frame": np.arange(n),
            "intensity": np.ones(n),
        }
    )

    raw_a, raw_b, _ = render_two_images(df_a, df_b, pixel_size_nm=5.0, gaussian_sigma_nm=0.0)
    blur_a, blur_b, _ = render_two_images(df_a, df_b, pixel_size_nm=5.0, gaussian_sigma_nm=5.0)

    assert not np.allclose(raw_a, blur_a)
    assert not np.allclose(raw_b, blur_b)
    assert np.isclose(np.sum(raw_a), np.sum(blur_a), rtol=0.02)

    freqs_raw, frc_raw = compute_frc_curve(raw_a - raw_a.mean(), raw_b - raw_b.mean())
    freqs_blur, frc_blur = compute_frc_curve(blur_a - blur_a.mean(), blur_b - blur_b.mean())
    cutoff_raw = find_cutoff_frequency_robust(freqs_raw, frc_raw, threshold=1 / 7, smooth_bins=5, nyquist_guard=0.49)
    cutoff_blur = find_cutoff_frequency_robust(freqs_blur, frc_blur, threshold=1 / 7, smooth_bins=5, nyquist_guard=0.49)

    # This is a diagnostic expectation: common filtering can preserve the final
    # cutoff even though rendered pixels and spectra changed.
    if cutoff_raw is not None and cutoff_blur is not None:
        assert abs(cutoff_raw - cutoff_blur) < 0.08


def test_apodization_sigma_changes_nontrivial_window_array():
    mask = np.zeros((31, 31), dtype=bool)
    mask[6:25, 6:25] = True

    hard = _apodize_mask(mask, sigma_px=0.0)
    soft_1 = _apodize_mask(mask, sigma_px=1.0)
    soft_5 = _apodize_mask(mask, sigma_px=5.0)

    assert set(np.unique(hard)).issubset({0.0, 1.0})
    assert not np.allclose(soft_1, soft_5)
    assert 0.0 < soft_5[5, 15] < soft_1[6, 15]


def test_all_ones_rectangular_roi_currently_produces_no_taper():
    mask = np.ones((21, 21), dtype=bool)

    window = _apodize_mask(mask, sigma_px=8.0)

    # Documents current behavior only: a crop whose mask is all ones has no
    # outside-zero support, so Gaussian mask apodization leaves it untapered.
    assert np.allclose(window, np.ones_like(window))


def test_gui_and_backend_apodization_paths_are_explicitly_reported():
    gui_meta = frc_parameter_metadata(
        pixel_size_nm=5.0,
        gaussian_sigma_nm=5.0,
        roi_apod_sigma_px=4.0,
        threshold=1 / 7,
        split_mode="odd_even",
        roi_source="current_roi_layer",
        random_seed=0,
        roi_apod_sigma_source="gui_control",
    )
    backend_meta = {
        "roi_apod_sigma_px_effective": BACKEND_ROI_APOD_SIGMA_PX,
        "roi_apod_sigma_source": BACKEND_ROI_APOD_SIGMA_SOURCE,
    }

    assert gui_meta["roi_apod_sigma_px"] == 4.0
    assert gui_meta["roi_apod_sigma_source"] == "gui_control"
    assert backend_meta["roi_apod_sigma_px_effective"] == 2.0
    assert backend_meta["roi_apod_sigma_source"] == "backend_hardcoded_default"


def test_export_parameter_metadata_contains_required_diagnostic_fields():
    meta = frc_parameter_metadata(
        pixel_size_nm=5.0,
        gaussian_sigma_nm=8.0,
        roi_apod_sigma_px=2.0,
        threshold=1 / 7,
        split_mode="random_blocks",
        roi_source="current_roi_layer",
        random_seed=42,
        roi_apod_sigma_source="gui_control",
    )

    assert meta == {
        "pixel_size_nm": 5.0,
        "gaussian_sigma_nm": 8.0,
        "gaussian_sigma_px": 1.6,
        "roi_apod_sigma_px": 2.0,
        "threshold": 1 / 7,
        "split_mode": "random_blocks",
        "roi_source": "current_roi_layer",
        "random_seed": 42,
        "roi_apod_sigma_source": "gui_control",
    }


def test_compute_frc_plot_notes_report_apodization_sigma():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(32, 32)).astype(np.float32)
    b = a + 0.1 * rng.normal(size=(32, 32)).astype(np.float32)
    roi = np.zeros((32, 32), dtype=bool)
    roi[4:28, 4:28] = True

    plot = compute_frc_plot_data(a, b, pixel_size_nm=5.0, roi_mask=roi, apod_sigma_px=3.5)

    assert "apod_sigma_px=3.5" in plot.notes
