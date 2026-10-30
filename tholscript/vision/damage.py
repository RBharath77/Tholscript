"""Physical-looking degradation of a clean inscription render.

A model trained on clean vector glyphs scores 99% and is useless on a
rain-worn hero stone.  Everything in :func:`degrade` is a *physical* process
you can point at on a real slab:

======================  =====================================================
stone texture           low-frequency value noise = granite grain / weathering
lighting                directional gradient = raking sunlight or torch light
chipping                soft-edged blobs removed = flaking / tool damage
cracks                  random-walk fractures = thermal stress
lichen                  smooth pale blobs = biological growth
partial loss            a whole band of the line worn away = erosion
erosion / dilation      the cutter's tool marks, and ink spread in the groove
perspective             the photo was not taken square to the slab
blur + noise            camera and scanner
======================  =====================================================

Each effect is switched on by probability and its strength is scaled by
``DamageConfig.level`` so we can measure an accuracy-vs-damage curve.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tholscript.config import DamageConfig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Accept None / int / random.Random / Generator and always return a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, random.Random):
        return np.random.default_rng(seed.randint(0, 2 ** 31 - 1))
    return np.random.default_rng(seed)


def _value_noise(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Bilinearly upsampled random grid -> smooth 0..1 field (fBm building block)."""
    grid = rng.random((cells, cells), dtype=np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def fractal_noise(h: int, w: int, rng, octaves: int = 4) -> np.ndarray:
    """Multi-octave value noise in 0..1; `rng` may be a Generator or random.Random."""
    rng = _rng(rng)
    out = np.zeros((h, w), np.float32)
    amp, total, cells = 1.0, 0.0, 4
    for _ in range(octaves):
        out += amp * _value_noise(h, w, cells, rng)
        total += amp
        amp *= 0.5
        cells *= 2
    return out / max(total, 1e-6)


def _soft_blob_mask(h: int, w: int, n: int, rng: np.random.Generator,
                    r_lo: float = 0.05, r_hi: float = 0.18) -> np.ndarray:
    """n soft-edged blobs, value 1 inside -> 0 outside."""
    mask = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        r = rng.uniform(r_lo, r_hi) * min(h, w) * rng.uniform(0.7, 2.0)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.4) ** 2)
        mask = np.maximum(mask, np.clip(1.0 - (d / max(r, 1.0)) ** 2, 0, 1))
    return mask


def _crack(img: np.ndarray, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Random-walk fractures that thin the ink as they cross it."""
    h, w = img.shape
    out = img.copy()
    for _ in range(n):
        length = rng.integers(int(0.4 * h), int(1.6 * h))
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(0.8, 3.0)
        pts = []
        for _step in range(length):
            ang += rng.normal(0, 0.22)
            x += np.cos(ang)
            y += np.sin(ang)
            pts.append((int(x), int(y)))
        if len(pts) < 2:
            continue
        crack = np.zeros((h, w), np.uint8)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            cv2.line(crack, (x0, y0), (x1, y1), 255, max(1, int(round(width))))
        crack = cv2.GaussianBlur(crack, (3, 3), 0).astype(np.float32) / 255.0
        out = out * (1.0 - 0.92 * crack)
    return out


@dataclass
class DamageTrace:
    """Which effects actually fired - shown in the web UI next to the image."""
    applied: List[str]

    def describe(self) -> str:
        return ", ".join(self.applied) if self.applied else "clean render (no damage)"


# --------------------------------------------------------------------------- #
# engraved-letterform randomisation (closes the print->chisel gap)
# --------------------------------------------------------------------------- #
def elastic(img: np.ndarray, rng: np.random.Generator, alpha: float = 3.0,
            sigma: float = 3.0) -> np.ndarray:
    """Classic elastic distortion: hand-cut letters wobble, print does not."""
    h, w = img.shape
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=0)


def thin_strokes(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Chisel grooves are thinner and rougher than font strokes."""
    k = int(rng.integers(1, 3))
    out = cv2.erode(img, np.ones((k, k), np.uint8))
    rough = rng.normal(0, 0.18, img.shape).astype(np.float32)
    return np.clip(out * (0.85 + 0.3 * (rough > 0)), 0, 1)


def lowres_pass(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a small field photo / thumbnail: down then back up."""
    h, w = img.shape
    f = rng.uniform(0.30, 0.55)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpegish(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(45, 75))
    ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1) * 255).astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def degrade(img: np.ndarray, cfg: DamageConfig = DamageConfig(),
            seed: Optional[int] = None) -> Tuple[np.ndarray, DamageTrace]:
    """Degrade an *ink map* (float32 HxW, 1 = ink, 0 = stone).

    Returns the degraded ink map: every effect here removes or thins ink, it
    never paints the stone.  Lighting and the stone surface itself are applied
    once per photo by :func:`compose_photo`, both during training and at
    inference, so the two always agree.
    """
    rng = _rng(seed)
    img = np.clip(img.astype(np.float32), 0, 1)
    h, w = img.shape
    applied: List[str] = []
    p = cfg.scale

    if rng.random() < p("stone_texture"):
        tex = fractal_noise(h, w, rng, octaves=5)
        # weathering eats ink non-uniformly: the cutter's groove is shallower
        img = img * (0.30 + 0.85 * tex)
        applied.append("stone texture")

    if rng.random() < p("chipping"):
        n = int(rng.integers(3, 9))
        mask = _soft_blob_mask(h, w, n, rng, 0.05, 0.20)
        img = img * (1.0 - 0.95 * mask)
        applied.append(f"chipping x{n}")

    if rng.random() < p("crack"):
        n = int(rng.integers(1, 3 + cfg.level))
        img = _crack(img, rng, n)
        applied.append(f"cracks x{n}")

    if rng.random() < p("lichen"):
        n = int(rng.integers(1, 4))
        blob = _soft_blob_mask(h, w, n, rng, 0.08, 0.22)
        img = img * (1.0 - 0.85 * blob) + 0.22 * blob
        applied.append("lichen")

    if rng.random() < p("partial_loss"):
        # a contiguous band of the line is simply gone, with soft edges
        band_w = max(2, int(rng.uniform(0.06, 0.22) * w))
        x0 = int(rng.uniform(0, max(1, w - band_w)))
        taper = max(2, int(band_w * 0.35))
        fade = np.ones(w, np.float32)
        fade[x0:x0 + band_w] = 0.0
        if x0 - taper >= 0:
            fade[x0 - taper:x0] = np.linspace(1, 0, taper, dtype=np.float32)
        if x0 + band_w + taper <= w:
            fade[x0 + band_w:x0 + band_w + taper] = np.linspace(0, 1, taper,
                                                                 dtype=np.float32)
        img = img * fade[None, :]
        applied.append("worn band")

    if rng.random() < p("erode"):
        k = int(rng.integers(1, 2 + cfg.level))
        if k > 0:
            img = cv2.erode(img, np.ones((k + 1, k + 1), np.uint8))
            applied.append(f"erosion k={k}")
    elif rng.random() < p("dilate"):
        k = int(rng.integers(1, 2))
        img = cv2.dilate(img, np.ones((k + 1, k + 1), np.uint8))
        applied.append(f"ink spread k={k}")

    return np.clip(img, 0, 1).astype(np.float32), DamageTrace(applied)


def compose_photo(stone_base: np.ndarray, ink: np.ndarray,
                  cfg: DamageConfig = DamageConfig(),
                  seed: Optional[int] = None,
                  groove: float = 0.10) -> Tuple[np.ndarray, DamageTrace]:
    """Merge an ink map onto a stone surface and add the photo-level effects:
    perspective, raking light, focus blur and sensor noise.

    ``groove`` is the darkness of the cut relative to the surrounding stone.
    """
    rng = _rng(seed)
    h, w = ink.shape
    applied: List[str] = []
    p = cfg.scale

    photo = stone_base * (1.0 - groove - ink * (0.88 - groove)) + ink * groove
    # a thin halo where the tool bruised the stone around the groove
    halo = cv2.GaussianBlur(ink, (7, 7), 2.0) - ink
    photo = photo - 0.10 * np.clip(halo, 0, 1)

    if rng.random() < p("perspective"):
        d = rng.uniform(0.02, 0.08)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [rng.uniform(0, d) * w, rng.uniform(0, d) * h],
            [w - rng.uniform(0, d) * w, rng.uniform(0, d) * h],
            [w - rng.uniform(0, d) * w, h - rng.uniform(0, d) * h],
            [rng.uniform(0, d) * w, h - rng.uniform(0, d) * h],
        ])
        photo = cv2.warpPerspective(photo, cv2.getPerspectiveTransform(src, dst),
                                    (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        applied.append("perspective")

    if rng.random() < p("lighting"):
        gx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        ramp = gx if rng.random() < 0.5 else gy
        strength = rng.uniform(0.15, 0.45)      # mild: real slabs stay legible
        photo = photo * (1.0 - strength * (ramp - 0.4))
        applied.append("raking light")

    if rng.random() < p("blur"):
        k = int(rng.choice([3, 5]))
        s = rng.uniform(0.4, 1.1)
        photo = cv2.GaussianBlur(photo, (k, k), s)
        applied.append("blur")

    if rng.random() < p("noise"):
        photo = photo + rng.normal(0, rng.uniform(0.015, 0.05),
                                   photo.shape).astype(np.float32)
        applied.append("sensor noise")

    return np.clip(photo, 0, 1).astype(np.float32), DamageTrace(applied)


def render_stone_photo(img: np.ndarray, seed: Optional[int] = None,
                       size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Turn an already-degraded line strip into a full slab photo for the demo.

    Adds a stone background, vignette, a slight rotation and JPEG-ish artefacts,
    plus a border so the line segmentation stage has something real to find.
    """
    rng = _rng(seed)
    h, w = img.shape
    if size is not None:
        tw, th = size
        scale = min(tw / w, th / h)
        img = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))))
        h, w = img.shape
    pad_y, pad_x = int(0.75 * h), int(0.25 * w)
    H, W = h + 2 * pad_y, w + 2 * pad_x
    bg = fractal_noise(H, W, rng, octaves=5) * 0.35 + 0.45
    bg += rng.normal(0, 0.03, (H, W)).astype(np.float32)
    canvas = bg.copy()
    y0, x0 = pad_y, pad_x
    canvas[y0:y0 + h, x0:x0 + w] = np.minimum(canvas[y0:y0 + h, x0:x0 + w] * 0.9,
                                              img * 0.95)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    canvas = canvas * (1.0 - 0.30 * np.clip(d - 0.55, 0, 1) ** 1.5)
    # slight rotation of the whole slab
    ang = rng.uniform(-3.0, 3.0)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    canvas = cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(canvas, 0, 1).astype(np.float32)
