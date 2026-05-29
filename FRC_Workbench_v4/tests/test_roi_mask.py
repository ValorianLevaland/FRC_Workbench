import numpy as np
from frc_workbench.core.frc_backend import ROIRecord


def test_roi_mask_orientation_smoke():
    # A rectangle polygon in (row, col): rows ~1..3, cols ~2..4
    rec = ROIRecord(
        version=1,
        created_at="2020-01-01T00:00:00Z",
        nm_per_px=10.0,
        image_shape=(6, 8),
        polygon_px=[[1, 2], [1, 5], [4, 5], [4, 2]],
        source={"basename": "x", "recon_tag": "t", "sha256": ""},
    )
    m = rec.mask()
    assert m.shape == (6, 8)
    # interior points should be inside
    assert bool(m[2, 3]) is True
    assert bool(m[3, 4]) is True
    # obvious outside points
    assert bool(m[0, 0]) is False
    assert bool(m[5, 7]) is False
