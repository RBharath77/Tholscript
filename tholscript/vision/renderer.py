"""Synthetic inscription renderer.

Everything the models are trained on, and everything the demo photos are made
of, comes out of here.  Three levels of granularity:

``render_glyph``    one grapheme cluster, clean, tight bbox  -> glyph CNN
``render_line``     one line of text as a stone photo strip  -> CRNN / CTC
``render_canvas``   a whole slab photo with several lines    -> the web demo

Clusters are composited *sequentially* (ink bbox for each cluster, advance from
``font.getlength``) rather than shaping a whole string at once.  That is what
gives us exact per-character ground-truth boxes, which the pipeline uses for
localisation and for the accuracy-vs-damage report.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tholscript.config import (
    CHAR_SPACING_RANGE, FONT_SIZE_RANGE, FONTS, IMAGE_HEIGHT, MAX_LINE_WIDTH,
    DamageConfig,
)
from tholscript.data.inventory import clusterize
from tholscript.vision.damage import (DamageTrace, compose_photo, degrade,
                                  elastic, fractal_noise, jpegish, lowres_pass,
                                  thin_strokes)

# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
_FONT_CACHE: Dict[Tuple[str, str, int], ImageFont.FreeTypeFont] = {}


def get_font(script: str, style: str, size: int) -> ImageFont.FreeTypeFont:
    key = (script, style, size)
    if key not in _FONT_CACHE:
        path = FONTS[script][style]
        _FONT_CACHE[key] = ImageFont.truetype(str(path), size)
    return _FONT_CACHE[key]


def _styles_for(script: str) -> List[str]:
    return list(FONTS[script].keys())


# --------------------------------------------------------------------------- #
# data classes
# --------------------------------------------------------------------------- #
@dataclass
class LineSample:
    image: np.ndarray              # float32 HxW photo strip (what the model sees)
    ink: np.ndarray                # degraded ink map, 1 = ink
    clean: np.ndarray              # undamaged render, for reporting
    text: str                      # full line text
    chars: List = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    trace: Optional[DamageTrace] = None
    script: str = "tamil"


@dataclass
class CanvasSample:
    image: np.ndarray              # float32 HxW, 0..1, a "photo" of the slab
    lines: List[LineSample]
    line_boxes: List[Tuple[int, int, int, int]]   # (x0, y0, x1, y1) per line
    script: str


# --------------------------------------------------------------------------- #
# glyph rendering
# --------------------------------------------------------------------------- #
def render_glyph(cluster: str, script: str = "tamil", size: int = 96,
                 style: Optional[str] = None, rng: Optional[random.Random] = None,
                 stroke: int = 0) -> Optional[np.ndarray]:
    """Render one grapheme cluster to a tight float32 ink map, or None if blank."""
    rng = rng or random.Random(0)
    style = style or rng.choice(_styles_for(script))
    font = get_font(script, style, size)
    bbox = font.getbbox(cluster)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    pad = 4 + stroke
    w, h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), cluster, font=font, fill=255,
                             stroke_width=stroke, stroke_fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


def glyph_patch(cluster: str, script: str = "tamil", out_size: int = 56,
                rng: Optional[random.Random] = None, stroke: int = 0,
                jitter: bool = True) -> Optional[np.ndarray]:
    """Cluster -> normalised out_size x out_size patch with a little pose jitter."""
    rng = rng or random.Random(0)
    size = rng.randint(88, 128) if jitter else 104
    img = render_glyph(cluster, script=script, size=size, rng=rng, stroke=stroke)
    if img is None:
        return None
    h, w = img.shape
    if jitter:
        scale = rng.uniform(0.72, 1.15)
        nw, nh = max(4, int(w * scale)), max(4, int(h * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        ang = rng.uniform(-9, 9)
        M = cv2.getRotationMatrix2D((nw / 2, nh / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR)
    else:
        nw, nh = w, h

    canvas = np.zeros((out_size, out_size), np.float32)
    m = 4
    s = min((out_size - 2 * m) / max(nw, nh), 1.0)
    fw, fh = max(1, int(nw * s)), max(1, int(nh * s))
    small = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
    dy = (out_size - fh) // 2 + (rng.randint(-3, 3) if jitter else 0)
    dx = (out_size - fw) // 2 + (rng.randint(-3, 3) if jitter else 0)
    y0, y1 = max(0, dy), min(out_size, dy + fh)
    x0, x1 = max(0, dx), min(out_size, dx + fw)
    sy0, sy1 = max(0, -dy), max(0, -dy) + (y1 - y0)
    sx0, sx1 = max(0, -dx), max(0, -dx) + (x1 - x0)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = small[sy0:sy1, sx0:sx1]
    return canvas


@dataclass
class CharBox:
    text: str
    x0: int
    x1: int
    y0: int
    y1: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

# --------------------------------------------------------------------------- #
# line rendering
# --------------------------------------------------------------------------- #
def _compose_line(clean: np.ndarray, cfg: DamageConfig, rng: random.Random,
                  seed: int, camera: bool = True) -> Tuple[np.ndarray, np.ndarray, DamageTrace]:
    extra = []
    ngr = np.random.default_rng(seed)
    if rng.random() < 0.60:                       # hand-cut wobble
        clean = elastic(clean, ngr, alpha=rng.uniform(1.5, 3.5))
        extra.append("elastic wobble")
    if rng.random() < 0.50:                       # chisel = thin rough strokes
        clean = thin_strokes(clean, ngr)
        extra.append("chisel strokes")
    ink, ink_trace = degrade(clean, cfg, seed=seed)
    stone = fractal_noise(clean.shape[0], clean.shape[1], rng, octaves=4) * 0.30 + 0.50
    photo, photo_trace = compose_photo(stone, ink, cfg, seed=seed + 1)
    if camera:
        if rng.random() < 0.50:                   # small field photo / thumbnail
            photo = lowres_pass(photo, np.random.default_rng(seed + 2))
            extra.append("low-res photo")
        if rng.random() < 0.60:
            photo = jpegish(photo, np.random.default_rng(seed + 3))
            extra.append("jpeg artefacts")
    trace = DamageTrace(ink_trace.applied + photo_trace.applied + extra)
    return photo, ink, trace


def render_line(text: str, script: str = "tamil", cfg: Optional[DamageConfig] = None,
                rng: Optional[random.Random] = None, degrade_it: bool = True,
                seed: Optional[int] = None, camera: bool = True) -> LineSample:
    """Render a whole line onto a photo strip, with per-character boxes."""
    rng = rng or random.Random(0)
    cfg = cfg or DamageConfig()
    style = rng.choice(_styles_for(script))
    size = rng.randint(*FONT_SIZE_RANGE)
    font = get_font(script, style, size)
    spacing = rng.randint(*CHAR_SPACING_RANGE)
    baseline = int(IMAGE_HEIGHT * 0.78)
    seed = seed if seed is not None else rng.randint(0, 2 ** 31 - 1)

    chars = []
    layers = []
    pen = 10.0
    for ch in clusterize(text, script):
        bbox = font.getbbox(ch)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        if bx1 - bx0 < 1:
            pen += font.getlength(ch) + spacing
            continue
        w = bx1 - bx0
        h = by1 - by0
        tmp = Image.new("L", (w + 8, h + 8), 0)
        ImageDraw.Draw(tmp).text((4 - bx0, 4 - by0), ch, font=font, fill=255)
        arr = np.asarray(tmp, dtype=np.float32) / 255.0
        px = int(round(pen)) + bx0 - 4
        py = baseline - by1 - 4
        layers.append((px, py, arr))
        chars.append(CharBox(text=ch, x0=px, x1=px + w, y0=max(0, py), y1=py + h))
        pen += font.getlength(ch) + spacing

    if not layers:
        empty = np.zeros((IMAGE_HEIGHT, 32), np.float32)
        return LineSample(image=empty, ink=empty, clean=empty, text=text,
                          chars=[], words=text.split(), script=script)

    width = min(MAX_LINE_WIDTH, int(pen) + 16)
    clean = np.zeros((IMAGE_HEIGHT, width), np.float32)
    for px, py, arr in layers:
        h, w = arr.shape
        y0, y1 = max(0, py), min(IMAGE_HEIGHT, py + h)
        x0, x1 = max(0, px), min(width, px + w)
        if y1 <= y0 or x1 <= x0:
            continue
        clean[y0:y1, x0:x1] = np.maximum(clean[y0:y1, x0:x1],
                                         arr[y0 - py:y1 - py, x0 - px:x1 - px])

    if degrade_it:
        photo, ink, trace = _compose_line(clean, cfg, rng, seed, camera=camera)
    else:
        photo = np.clip(0.55 - 0.45 * clean, 0, 1).astype(np.float32)
        ink = clean.copy()
        trace = DamageTrace([])
    return LineSample(image=photo, ink=ink, clean=clean, text=text, chars=chars,
                      words=text.split(), trace=trace, script=script)


# --------------------------------------------------------------------------- #
# full-slab rendering (the demo photos)
# --------------------------------------------------------------------------- #
def render_canvas(lines: Sequence[str], script: str = "tamil",
                  cfg: Optional[DamageConfig] = None,
                  rng: Optional[random.Random] = None,
                  gap: Optional[int] = None) -> CanvasSample:
    """Compose several damaged lines into one slab photo with GT line boxes."""
    rng = rng or random.Random(0)
    cfg = cfg or DamageConfig()
    seed = rng.randint(0, 2 ** 31 - 1)
    # the showcase slab keeps the chisel/elastic variation but not the
    # bad-camera passes (low-res / jpeg): those belong to TRAINING only
    samples = [render_line(t, script=script, cfg=cfg, rng=rng, degrade_it=True,
                           seed=rng.randint(0, 2 ** 31 - 1), camera=False)
               for t in lines]
    width = max(s.clean.shape[1] for s in samples)
    gap = gap if gap is not None else int(IMAGE_HEIGHT * 0.45)
    top = int(IMAGE_HEIGHT * 0.7)

    total_h = top + sum(s.clean.shape[0] for s in samples) + gap * len(samples) + top
    ink_layer = np.zeros((total_h, width + 2 * top), np.float32)

    y = top
    line_boxes: List[Tuple[int, int, int, int]] = []
    degraded_traces = []
    for s in samples:
        h, w = s.clean.shape
        x0 = top
        ink_layer[y:y + h, x0:x0 + w] = np.maximum(ink_layer[y:y + h, x0:x0 + w], s.ink)
        line_boxes.append((x0, y, x0 + w, y + h))
        y += h + gap

    rng2 = np.random.default_rng(seed)
    stone = fractal_noise(total_h, width + 2 * top, rng2, octaves=5) * 0.32 + 0.56
    photo, trace = compose_photo(stone, ink_layer, cfg, seed=seed + 7)

    # gentle vignette for the whole slab
    H, W = photo.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    photo = np.clip(photo * (1.0 - 0.18 * np.clip(d - 0.6, 0, 1) ** 1.4), 0, 1)
    ang = rng.uniform(-1.2, 1.2)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    photo = cv2.warpAffine(photo, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return CanvasSample(image=np.clip(photo, 0, 1).astype(np.float32),
                        lines=samples, line_boxes=line_boxes, script=script)


def save_png(img: np.ndarray, path) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1) * 255).astype(np.uint8))



if __name__ == "__main__":
    from tholscript.data.lexicon import sample_brahmi_line, sample_tamil_line

    rng = random.Random(3)
    for script, sampler in (("tamil", sample_tamil_line), ("brahmi", sample_brahmi_line)):
        s = render_line(sampler(rng).text, script=script,
                        cfg=DamageConfig(level=2), rng=rng)
        print(f"{script}: {s.text[:40]!r} shape={s.image.shape} "
              f"chars={len(s.chars)} damage=[{s.trace.describe()}]")
    c = render_canvas([sample_tamil_line(rng).text for _ in range(3)], "tamil",
                      cfg=DamageConfig(level=3), rng=rng)
    print("canvas", c.image.shape, "line boxes", len(c.line_boxes))
    save_png(c.image, "artifacts/_smoke_canvas.png")
