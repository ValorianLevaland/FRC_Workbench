# batch_frc_cli.py
from __future__ import annotations
import argparse
import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

from batch_frc_backend import (
    process_single_csv,
    find_odd_even_pairs,
    process_tif_pair,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch FRC estimator + RSP/RSE/FRC maps for odd/even TIF pairs.")
    # Common
    p.add_argument("--root", type=str, required=True, help="Root directory to search (recursively).")
    p.add_argument("--out", type=str, required=True, help="Output root directory (mirrors subfolders).")
    p.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto; 1=serial).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    # CSV mode
    p.add_argument("--do-csv", action="store_true", help="Process ThunderSTORM-style CSVs to overall FRC.")
    p.add_argument("--glob", type=str, default="*customed_title.csv",
                   help="Glob pattern for CSV file names (default: *customed_title.csv).")
    p.add_argument("--include", type=str, default="",
                   help="Comma-separated substrings that MUST appear in the CSV filename (case-insensitive).")
    p.add_argument("--exclude", type=str, default="",
                   help="Comma-separated substrings that MUST NOT appear in the CSV filename (case-insensitive).")
    p.add_argument("--method", type=str, default="odd_even", choices=["odd_even", "random_blocks"],
                   help="Splitting method for CSV-based FRC: odd_even or random_blocks.")
    p.add_argument("--block-size-frames", type=int, default=500,
                   help="Block size in frames (used by random_blocks).")
    p.add_argument("--pixel-size-nm", type=float, default=10.0, help="Pixel size for rendering (nm/pixel).")
    p.add_argument("--gaussian-sigma-nm", type=float, default=15.0, help="Gaussian smoothing sigma (nm).")
    p.add_argument("--weight-mode", type=str, default="ones", choices=["ones", "intensity"],
                   help="Pixel weights: 'ones' or 'intensity'.")
    p.add_argument("--threshold", type=float, default=1.0/7.0, help="FRC threshold (default: 1/7).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for random_blocks.")
    p.add_argument("--save-debug", action="store_true", help="Print per-file debug info to console.")

    # Pair (odd/even) TIF mode
    p.add_argument("--do-pairs", action="store_true", help="Process odd/even reconstructed TIF pairs to RSP/RSE/FRC maps.")
    p.add_argument("--odd-glob", type=str, default="*_odd.tif", help="Glob for odd TIFs (default: *_odd.tif).")
    p.add_argument("--even-glob", type=str, default="*_even.tif", help="Glob for even TIFs (default: *_even.tif).")
    p.add_argument("--rsp-window", type=int, default=21, help="Window size (odd) for RSP/RSE local maps.")
    p.add_argument("--squirrel-sigma-px", type=float, default=1.5, help="Gaussian sigma (pixels) before SQUIRREL metrics.")
    p.add_argument("--squirrel-auto-sigma", action="store_true", help="Auto-pick sigma that maximizes global RSP.")
    p.add_argument("--frc-tile", type=int, default=64, help="Tile size (pixels) for FRC map.")
    p.add_argument("--frc-stride", type=int, default=64, help="Stride (pixels) between tiles for FRC map.")
    p.add_argument("--pair-threshold", type=float, default=1.0/7.0, help="FRC threshold for pair-based FRC map.")
    return p.parse_args()

def find_csv_files(root: Path, pattern: str, includes: List[str], excludes: List[str]) -> List[Path]:
    root = root.resolve()
    files: List[Path] = []
    for p in root.rglob("*.csv"):
        name = p.name
        if not fnmatch.fnmatch(name, pattern):
            continue
        lname = name.lower()
        if includes and not all(s in lname for s in includes):
            continue
        if any(s in lname for s in excludes):
            continue
        files.append(p)
    return sorted(files)

def _mirror_out_dir(out_root: Path, root: Path, fpath: Path) -> Path:
    rel = fpath.parent.relative_to(root)
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    include = [s.strip().lower() for s in args.include.split(",") if s.strip()]
    exclude = [s.strip().lower() for s in args.exclude.split(",") if s.strip()]

    tasks: List[Tuple[str, Tuple, Dict]] = []  # (kind, args, kwargs)
    csv_files: List[Path] = []
    pair_triples: List[Tuple[Path, Path, str]] = []

    if args.do_csv:
        csv_files = find_csv_files(root, args.glob, include, exclude)
        for f in csv_files:
            out_dir = _mirror_out_dir(out_root, root, f)
            tasks.append(("csv", (f, out_dir), dict(
                method=args.method,
                block_size_frames=args.block_size_frames,
                pixel_size_nm=args.pixel_size_nm,
                gaussian_sigma_nm=args.gaussian_sigma_nm,
                weight_mode=args.weight_mode,
                threshold=args.threshold,
                seed=args.seed,
                overwrite=args.overwrite
            )))

    if args.do_pairs:
        # We pair within each directory using _odd/_even naming (case-insensitive)
        # We ignore --odd-glob/--even-glob for pairing logic but they still help your dataset organization.
        pair_triples = find_odd_even_pairs(root)
        for odd_path, even_path, base in pair_triples:
            out_dir = _mirror_out_dir(out_root, root, odd_path)
            tasks.append(("pair", (odd_path, even_path, out_dir), dict(
                pixel_size_nm=args.pixel_size_nm,
                squirrel_window=args.rsp_window,
                squirrel_sigma_px=args.squirrel_sigma_px,
                squirrel_auto_sigma=args.squirrel_auto_sigma,
                frc_threshold=args.pair_threshold,
                frc_tile=args.frc_tile,
                frc_stride=args.frc_stride,
                overwrite=args.overwrite
            )))

    if not tasks:
        print("No work to do. (Enable --do-csv and/or --do-pairs)")
        return 1

    # Parallel execution
    workers = None if args.workers == 0 else max(1, args.workers)
    results = []
    errors = []

    def _submit(exe, kind, a, kw):
        if kind == "csv":
            return exe.submit(process_single_csv, *a, **kw)
        else:
            return exe.submit(process_tif_pair, *a, **kw)

    if workers == 1:
        # serial
        for kind, a, kw in tasks:
            try:
                res = process_single_csv(*a, **kw) if kind == "csv" else process_tif_pair(*a, **kw)
                results.append(res)
                if args.save_debug:
                    print(f"OK: {res.get('input_csv', res.get('odd_tif'))}")
            except Exception as e:
                errors.append((a, str(e)))
                print(f"ERROR: {a} -> {e}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            fut_to_task = { _submit(exe, kind, a, kw): (kind, a) for (kind, a, kw) in tasks }
            for fut in as_completed(fut_to_task):
                kind, a = fut_to_task[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    if args.save_debug:
                        key = res.get("input_csv", res.get("odd_tif"))
                        print(f"OK: {key}")
                except Exception as e:
                    errors.append((a, str(e)))
                    print(f"ERROR: {a} -> {e}")

    # Write batch summary
    if results:
        df_sum = pd.DataFrame(results)
        batch_csv = out_root / "batch_summary.csv"
        df_sum.to_csv(batch_csv, index=False)
        print(f"\nSaved batch summary: {batch_csv} ({len(df_sum)} rows)")

    if errors:
        print(f"\n{len(errors)} errors encountered.")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
