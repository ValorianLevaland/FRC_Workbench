from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .frc_backend import gaussian_blur


def phase_correlation_shift(a: np.ndarray, b: np.ndarray, upsample: int = 8) -> Tuple[float, float]:
    """Estimate subpixel shift (dy, dx) aligning b -> a using phase correlation."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("phase correlation requires same shape")
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-12)
    c = np.fft.ifft2(R).real

    # integer peak
    y0, x0 = np.unravel_index(np.argmax(c), c.shape)
    H, W = c.shape
    if y0 > H // 2:
        y0 -= H
    if x0 > W // 2:
        x0 -= W

    up = max(1, int(upsample))
    if up == 1:
        return float(y0), float(x0)

    # subpixel refine using a small patch and FFT zero-padding
    win = 7
    r = win // 2
    ys = np.arange(y0 - r, y0 + r + 1) % H
    xs = np.arange(x0 - r, x0 + r + 1) % W
    patch = c[np.ix_(ys, xs)]
    P = np.fft.fft2(patch)
    pad = win * up
    Ppad = np.zeros((pad, pad), dtype=np.complex64)
    ymid = pad // 2 - win // 2
    xmid = pad // 2 - win // 2
    Ppad[ymid:ymid + win, xmid:xmid + win] = np.fft.fftshift(P)
    upcorr = np.fft.ifft2(np.fft.ifftshift(Ppad)).real
    yy, xx = np.unravel_index(np.argmax(upcorr), upcorr.shape)
    dy = (yy - pad // 2) / float(up)
    dx = (xx - pad // 2) / float(up)
    return float(y0 + dy), float(x0 + dx)


def apply_shift_fourier(img: np.ndarray, shift_yx: Tuple[float, float]) -> np.ndarray:
    """Apply a subpixel shift using the Fourier shift theorem."""
    img = np.asarray(img, dtype=np.float32)
    dy, dx = map(float, shift_yx)
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.fft.ifft2(np.fft.fft2(img) * phase).real.astype(np.float32, copy=False)


def estimate_alpha_beta(sr: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit: ref ≈ alpha*sr + beta."""
    x = np.asarray(sr, dtype=np.float32).ravel()
    y = np.asarray(ref, dtype=np.float32).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return 1.0, 0.0
    x = x[m]
    y = y[m]
    vx = float(np.var(x))
    if vx <= 1e-12:
        return 1.0, float(np.mean(y) - np.mean(x))
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    alpha = cov / vx
    beta = float(np.mean(y) - alpha * np.mean(x))
    return float(alpha), float(beta)


def _local_rsp_rse(a: np.ndarray, b: np.ndarray, window: int = 21):
    """Local Pearson (RSP) and normalized error (RSE-like) between two images."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")
    k = int(window)
    if k < 3:
        raise ValueError("window must be >=3")
    if k % 2 == 0:
        k += 1

    def box_sum(img: np.ndarray, k: int) -> np.ndarray:
        r = k // 2
        pad = np.pad(img, ((r, r), (r, r)), mode="reflect")
        integ = pad.cumsum(axis=0).cumsum(axis=1)
        return (integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k])

    Sa = box_sum(a, k)
    Sb = box_sum(b, k)
    Saa = box_sum(a * a, k)
    Sbb = box_sum(b * b, k)
    Sab = box_sum(a * b, k)
    n = float(k * k)
    mu_a = Sa / n
    mu_b = Sb / n
    var_a = np.maximum(Saa / n - mu_a * mu_a, 0.0)
    var_b = np.maximum(Sbb / n - mu_b * mu_b, 0.0)
    std_a = np.sqrt(var_a)
    std_b = np.sqrt(var_b)
    cov_ab = (Sab / n) - (mu_a * mu_b)
    rsp = cov_ab / (std_a * std_b + 1e-12)
    rsp = np.clip(rsp, -1.0, 1.0)

    se = box_sum((a - b) ** 2, k)
    ref_energy = box_sum(a * a, k)
    num = np.sqrt(np.maximum(se, 0.0))
    den = np.sqrt(np.maximum(ref_energy, 0.0)) + 1e-12
    rse = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    aa = a - float(a.mean())
    bb = b - float(b.mean())
    global_rsp = float((aa * bb).sum() / (np.sqrt((aa * aa).sum()) * np.sqrt((bb * bb).sum()) + 1e-12))
    global_rse = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a * a)) + 1e-12))

    return rsp.astype(np.float32), rse.astype(np.float32), global_rsp, global_rse


def compute_squirrel_vs_reference(
    sr: np.ndarray,
    ref: np.ndarray,
    *,
    register: bool = True,
    upsample_factor: int = 8,
    sigma_min_px: float = 0.0,
    sigma_max_px: float = 3.0,
    sigma_steps: int = 13,
    optimize: str = "max_rsp",
    window: int = 21,
) -> Dict[str, Any]:
    """SQUIRREL-like workflow: align, degrade SR, fit intensity, output RSP/RSE maps.

    This is a *practical* Python-only approximation:

    - Optionally align SR to reference (phase correlation).
    - RSF model: blur(SR, sigma) then fit alpha/beta such that ref ≈ alpha*blur(SR)+beta.
    - Grid search sigma to maximize global RSP (Pearson) or minimize global RSE.
    - Return degraded SR and local RSP/RSE maps.

    Notes:
    - This implementation assumes SR and reference are already on the same pixel grid.
      (A future extension could resample the reference automatically.)
    """
    sr = np.asarray(sr, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if sr.shape != ref.shape:
        raise ValueError("SR and reference must have the same shape (this simplified workflow).")

    shift_yx = (0.0, 0.0)
    sr_aligned = sr
    if register:
        shift_yx = phase_correlation_shift(ref, sr, upsample=int(upsample_factor))
        sr_aligned = apply_shift_fourier(sr, shift_yx)

    sigmas = np.linspace(float(sigma_min_px), float(sigma_max_px), int(max(2, sigma_steps)))
    best = None
    best_score = None

    optimize = str(optimize).lower().strip()
    if optimize not in {"max_rsp", "min_rse"}:
        optimize = "max_rsp"

    for s in sigmas:
        sr_blur = gaussian_blur(sr_aligned, float(s))
        alpha, beta = estimate_alpha_beta(sr_blur, ref)
        pred = alpha * sr_blur + beta
        _, _, g_rsp, g_rse = _local_rsp_rse(ref, pred, window=int(window))
        if optimize == "min_rse":
            score = g_rse
            better = (best_score is None) or (score < best_score)
        else:
            score = g_rsp
            better = (best_score is None) or (score > best_score)
        if better:
            best_score = score
            best = (float(s), float(alpha), float(beta), float(g_rsp), float(g_rse), pred)

    if best is None:
        raise RuntimeError("Sigma search failed.")

    sigma, alpha, beta, g_rsp, g_rse, pred = best
    rsp_map, rse_map, _, _ = _local_rsp_rse(ref, pred, window=int(window))

    return dict(
        shift_yx=tuple(map(float, shift_yx)),
        sigma_px=float(sigma),
        alpha=float(alpha),
        beta=float(beta),
        global_rsp=float(g_rsp),
        global_rse=float(g_rse),
        degraded_sr=np.asarray(pred, dtype=np.float32),
        rsp_map=rsp_map,
        rse_map=rse_map,
    )
