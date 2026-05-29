# FRC Workbench (Napari + Matplotlib)

A Napari-first GUI application for **Fourier Ring Correlation (FRC)** analysis of single-molecule localization microscopy (SMLM) reconstructions.

It supports two complementary workflows:

- **Interactive / per-file analysis** (Napari viewer)
  - Load **ThunderSTORM/SMLM localization CSV** → split odd/even (or random blocks) → render SR halves
  - Draw ROI in Napari (polygon)
  - Compute **global FRC curve** (Matplotlib dock) + resolution estimate (nm)
  - Compute **tile-FRC resolution map** (nm) and **RSP/RSE maps**
  - Export reconstructions, maps, curves, ROI, and a run manifest

- **Batch analysis** (GUI, no terminal workflow)
  - Process multiple CSVs and/or odd/even TIFF pairs
  - Optional ROI reuse via saved ROI sidecars
  - Produces per-file outputs + batch summary tables

This repository is a **clean, self-contained build**: the numerical backend is included in `src/frc_workbench/core/` and the Napari GUI in `src/frc_workbench/gui/`.

---

## Installation (recommended)

Create/activate a dedicated environment, then install dependencies:

```bash
pip install -r requirements.txt
```

(Optional but recommended for development / editable install)

```bash
pip install -e .
```

---

## Run the GUI

### Option A — run as a module (most robust)

```bash
python -m frc_workbench
```

### Option B — run directly from the folder (no editable install)

```bash
python run_workbench.py
```

### Option B — run the launcher script

```bash
python scripts/launch_frc_workbench.py
```

### Option C — if installed as a package

```bash
frc-workbench
```

All options launch the **same GUI** (Napari viewer + dock widgets).

---

## Output files (exports)

When you click **Export**, the workbench creates an output folder containing:

- `recon_odd.tif`, `recon_even.tif`, `recon_sum.tif` (float32)
- `RSP_map.tif`, `RSE_map.tif` (float32)
- `FRC_tile_map_nm.tif` (float32)
- `FRC_curve.csv` + `FRC_curve.png`
- `ROI.roi.json` (+ optional `ROI_mask.tif`)
- `manifest.json` (parameters, software versions, hashes, timestamps)

---

## Notes / conventions

- **Resolution (nm)** is reported as: `pixel_size_nm / f_cutoff`, where `f_cutoff` is the FRC threshold crossing frequency in cycles/pixel.
- **Threshold** defaults to `1/7` (common for half-map FRC).
- ROI is stored in **reconstruction pixel coordinates** (row, col), plus the **nm/px** used when it was created, so it can be adapted if you change pixel size.

---

## Troubleshooting

- If Napari fails to start, ensure you installed the Qt backend:
  - `pip install "napari[pyqt5]"`
- For TIFF read/write, `tifffile` is strongly recommended.
- If SciPy is not available, the code falls back to slower pure-NumPy blurs (still correct, but less performant).

---

## License

This code is provided as-is for research use. Add your preferred license text in `LICENSE`.
