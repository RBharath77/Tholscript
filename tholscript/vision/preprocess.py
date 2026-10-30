"""Image enhancement + line segmentation used at inference *and* training.

Real inscription photos differ from the synthetic renders in three ways and
this module closes each of them:

1. **resolution** – field photos (and thumbnails like the ones people actually
   upload) are small; anything under ``UPSCALE_MIN`` is interpolated 3x first;
2. **contrast** – CLAHE equalises raking sunlight / shadow before background
   removal, and a morphological closing (not a Gaussian) estimates the stone,
   which survives the heavy texture of real granite;
3. **polarity** – cut letters read *darker* than the stone in some lights and
   *lighter* (sunlit groove) in others; both ink maps are computed and the one
   with the stronger text energy wins.

``enhance`` returns an ink map with 1 = letter strokes, the same object the
models were trained on.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

UPSCALE_MIN = 300


def to_gray(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    if a.max() > 1.5:                     # uint8 / 0-255 float input
        a /= 255.0
    if a.ndim == 3:
        a = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return a


def _clahe(g: np.ndarray) -> np.ndarray:
    g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    h, w = g8.shape
    tile = max(2, min(h, w) // 32)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    return cl.apply(g8).astype(np.float32) / 255.0


def _ink_map(g: np.ndarray, dark_strokes: bool) -> np.ndarray:
    """Hybrid background (gaussian + morphological close/open), signed difference."""
    h, w = g.shape
    k = max(9, int(min(h, w) * 0.09) | 1)
    if dark_strokes:
        gauss = cv2.GaussianBlur(g, (0, 0), sigmaX=k / 2.0)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = np.maximum(gauss, cv2.morphologyEx(g, cv2.MORPH_CLOSE, kern))
        diff = bg - g
    else:
        gauss = cv2.GaussianBlur(g, (0, 0), sigmaX=k / 2.0)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = np.minimum(gauss, cv2.morphologyEx(g, cv2.MORPH_OPEN, kern))
        diff = g - bg
    return np.clip(diff / np.maximum(bg, 0.08), 0, None).astype(np.float32)


def _normalise(ink: np.ndarray) -> np.ndarray:
    pos = ink[ink > 0]
    if pos.size == 0:
        return np.zeros_like(ink)
    lo = np.percentile(ink, 60)
    hi = np.percentile(pos, 97)
    if hi <= lo + 1e-6:
        hi = lo + 1e-3
    ink = np.clip((ink - lo) / (hi - lo), 0, 1)
    ink = cv2.medianBlur((ink * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    h, w = ink.shape
    m = max(3, int(min(h, w) * 0.015))
    ink[:m, :] = 0
    ink[-m:, :] = 0
    ink[:, :m] = 0
    ink[:, -m:] = 0
    return ink


def enhance(photo: np.ndarray, bg_sigma: Optional[float] = None,
          clahe: Optional[bool] = None) -> np.ndarray:
    """Raw photo (any size, any polarity) -> normalised ink map (1 = strokes).

    The map is returned at the *input* resolution (processing happens at an
    upscaled resolution internally), so callers can overlay boxes directly.
    """
    g = to_gray(photo)
    h, w = g.shape
    proc = g
    if min(h, w) < UPSCALE_MIN:
        f = int(np.ceil(UPSCALE_MIN / min(h, w)))
        proc = cv2.resize(g, (w * f, h * f), interpolation=cv2.INTER_CUBIC)
    # CLAHE rescues real field photos (granite grain, shadow) but amplifies
    # residual mottle on smooth synthetic stone; the high-frequency energy of
    # the photo tells the two apart (measured: synthetic ~0.02, real ~0.04)
    hf = float(np.abs(proc - cv2.GaussianBlur(proc, (0, 0), 2.0)).mean())
    use_clahe = (hf > 0.032) if clahe is None else clahe
    if use_clahe:
        proc = _clahe(proc)

    dark = _normalise(_ink_map(proc, dark_strokes=True))
    light = _normalise(_ink_map(proc, dark_strokes=False))

    def energy(m):
        v = np.sort(m.ravel())[::-1][:max(50, m.size // 50)]
        return float(v.sum())

    ink = dark if energy(dark) >= energy(light) else light
    if ink.shape != (h, w):
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA)
    return ink


def deskew(ink: np.ndarray, max_angle: float = 8.0) -> Tuple[np.ndarray, float]:
    """Rotate so the line bands are horizontal, using the ink's principal axis."""
    ys, xs = np.where(ink > 0.35)
    if len(xs) < 50:
        return ink, 0.0
    pts = np.stack([xs, ys], 1).astype(np.float32)
    _, (w_, h_), ang = cv2.minAreaRect(pts)
    if w_ < h_:
        ang = ang + 90
    ang = ((ang + 45) % 90) - 45
    if abs(ang) > max_angle:
        return ink, 0.0
    H, W = ink.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    out = cv2.warpAffine(ink, M, (W, H), flags=cv2.INTER_LINEAR)
    return out, ang


def segment_lines(ink: np.ndarray, min_height: int = 10, gap: int = 5,
                  thresh_frac: float = 0.10) -> List[Tuple[int, int]]:
    """Horizontal projection of *strong* ink -> (y0, y1) bands, top to bottom.

    Parameters are chosen to work both on synthetic slabs and on dense real
    inscriptions whose lines almost touch.
    """
    m = _stroke_only(ink)                       # drop blotchy stone residuals
    char_h = _char_height(m)                    # data-driven line-height prior
    strong = cv2.GaussianBlur((m > 0.25).sum(axis=1).astype(np.float32).reshape(-1, 1),
                              (1, 15), 0).ravel()
    if strong.max() <= 0:
        return []
    on = strong > strong.max() * min(thresh_frac, 0.08)
    bands: List[Tuple[int, int]] = []
    y0 = None
    for y, v in enumerate(on):
        if v and y0 is None:
            y0 = y
        elif not v and y0 is not None:
            bands.append((y0, y))
            y0 = None
    if y0 is not None:
        bands.append((y0, len(on)))
    gap = max(gap, char_h // 6)
    merged: List[List[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))
    merged = [b for b in merged if b[1] - b[0] >= max(min_height, char_h // 3)]
    merged = _split_tall(m, merged, char_h)
    # hysteresis: a real text line always contains a strong projection core,
    # while cracks / texture bands never do; then grow each kept band into its
    # faint edges (pulli dots, worn descenders)
    H = len(strong)
    out = []
    for a, b in merged:
        if strong[a:b].max() > 0.18 * strong.max() and (b - a) >= max(min_height, char_h // 3):
            grow = max(4, char_h // 4)
            ga = 0
            while a > 0 and ga < grow and strong[a - 1] > 0.06 * strong.max():
                a -= 1; ga += 1
            gb = 0
            while b < H and gb < grow and strong[b] > 0.06 * strong.max():
                b += 1; gb += 1
            out.append((a, b))
    return [(max(0, a - 2), min(H, b + 2)) for a, b in out]


def _char_height(m: np.ndarray) -> int:
    """How tall a text line is on this image, from the row profile's period.

    Damaged text fragments into blobs, so component sizes lie; but the
    *distance between line peaks* in the horizontal projection is stable in
    both synthetic slabs and real photos.
    """
    prof = (m > 0.25).sum(axis=1).astype(np.float32).reshape(-1, 1)
    prof = cv2.GaussianBlur(prof, (1, 21), 0).ravel()
    if prof.max() <= 0:
        return 24
    H = len(prof)
    thr = 0.6 * prof.max()
    peaks = []
    for y in range(1, H - 1):
        if prof[y] >= thr and prof[y] >= prof[y - 1] and prof[y] > prof[y + 1]:
            if not peaks or y - peaks[-1] > H // 25:
                peaks.append(y)
    spacings = [b - a for a, b in zip(peaks[:-1], peaks[1:])]
    if spacings:
        pitch = float(min(spacings))          # a missed peak inflates; min is safe
    else:
        pitch = H / 4.0
    return int(min(max(pitch / 1.25, 12), H / 3))


def _split_tall(m: np.ndarray, bands: List[List[int]], char_h: int) -> List[List[int]]:
    """Dense real inscriptions merge neighbouring lines; split any band that is
    much taller than a character line at its deepest projection valley."""
    profile = (m > 0.25).sum(axis=1).astype(np.float32).reshape(-1, 1)
    profile = cv2.GaussianBlur(profile, (1, 9), 0).ravel()
    out: List[List[int]] = []
    stack = [tuple(b) for b in bands]
    while stack:
        y0, y1 = stack.pop()
        hh = y1 - y0
        if hh <= 2.2 * char_h:
            out.append([y0, y1])
            continue
        lo, hi = y0 + hh // 5, y1 - hh // 5
        if hi <= lo:
            out.append([y0, y1])
            continue
        seg = profile[lo:hi]
        v = lo + int(np.argmin(seg))
        if profile[v] > 0.55 * seg.max():         # no real valley -> keep as one
            out.append([y0, y1])
            continue
        stack.append((y0, v))
        stack.append((v, y1))
    return sorted(out)


def _stroke_only(ink: np.ndarray) -> np.ndarray:
    """Morphological high-pass: wide stone blotches out, thin strokes stay."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blobs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    return np.clip(ink - 0.8 * blobs, 0, 1)


def extract_glyph_patch(ink_line: np.ndarray, x0: int, x1: int,
                        out_size: int = 56) -> np.ndarray:
    """Crop the ink between x0..x1, tight vertically, and fit to out_size."""
    crop = ink_line[:, max(0, x0):x1]
    ys, xs = np.where(crop > 0.3)
    if len(xs) == 0:
        return np.zeros((out_size, out_size), np.float32)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = min((out_size - 8) / max(h, w), 4.0)
    fw, fh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), np.float32)
    dy, dx = (out_size - fh) // 2, (out_size - fw) // 2
    canvas[dy:dy + fh, dx:dx + fw] = small
    return canvas
