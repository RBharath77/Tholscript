"""Model-free pipeline tests: rendering, enhancement, segmentation, LM, spellcheck.

These run in seconds and need no trained weights.  Weight-dependent behaviour
is covered by `python -m tholscript.scripts.cli evaluate` (see README).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tholscript.config import DamageConfig  # noqa: E402
from tholscript.data.inventory import clusterize  # noqa: E402
from tholscript.data.lexicon import sample_brahmi_line, sample_tamil_line  # noqa: E402
from tholscript.models.lm import build_lm  # noqa: E402
from tholscript.data.lexicon import ALL_WORDS  # noqa: E402
from tholscript.translation.spellcheck import SpellChecker  # noqa: E402
from tholscript.vision.preprocess import enhance, extract_glyph_patch, segment_lines  # noqa: E402
from tholscript.vision.renderer import render_canvas, render_line  # noqa: E402


def test_enhance_recovers_ink():
    rng = random.Random(4)
    s = render_line(sample_tamil_line(rng).text, "tamil",
                    cfg=DamageConfig(level=1), rng=rng)
    ink = enhance(s.image)
    a = (ink > 0.3).astype(np.uint8)
    b = (s.ink > 0.3).astype(np.uint8)
    iou = (a & b).sum() / max((a | b).sum(), 1)
    assert iou > 0.7, f"enhance IoU too low: {iou}"


def _check_segmentation(script, sampler, n, seed):
    rng = random.Random(seed)
    texts = [sampler(rng).text for _ in range(n)]
    canvas = render_canvas(texts, script, cfg=DamageConfig(level=2), rng=rng)
    ink = enhance(canvas.image)
    bands = segment_lines(ink)
    # every ground-truth line must be covered by a detected band
    for (x0, y0, x1, y1) in canvas.line_boxes:
        hit = [b for b in bands if b[1] > y0 and b[0] < y1]
        assert hit, f"line at y={y0}..{y1} not found (bands={bands})"
        ov = max(min(b[1], y1) - max(b[0], y0) for b in hit)
        assert ov > 0.45 * (y1 - y0), f"line at y={y0} covered only {ov}/{y1 - y0}"
    # ...and the detector must not hallucinate lots of extra bands
    assert len(bands) <= n + 2, f"{len(bands)} bands for {n} lines"


def test_line_segmentation_finds_every_line():
    _check_segmentation("tamil", sample_tamil_line, 4, 9)


def test_line_segmentation_brahmi():
    _check_segmentation("brahmi", sample_brahmi_line, 3, 2)


def test_clusterize_ground_truth_consistent():
    # every cluster must be renderable and present in the class list
    from tholscript.data.inventory import TAMIL_INVENTORY
    inv = set(TAMIL_INVENTORY)
    rng = random.Random(3)
    for _ in range(30):
        for c in clusterize(sample_tamil_line(rng).text, "tamil"):
            assert c in inv, c


def test_spellchecker_fixes_single_grapheme_errors():
    lm = build_lm([], ALL_WORDS)
    sc = SpellChecker(ALL_WORDS, lm)
    for good in ["கொடை", "மதுரை", "கோயில்", "நள்ளி"]:
        bad = good[:-1] + ("ழ" if good[-1] != "ழ" else "ள")
        r = sc.correct(bad)
        assert r.text == good, f"{bad} -> {r.text} (wanted {good})"


def test_lm_prefers_real_words():
    rng = random.Random(0)
    from tholscript.data.lexicon import sample_tamil_line
    lm = build_lm((sample_tamil_line(rng).text for _ in range(600)), ALL_WORDS)
    # the verb ending -த்தான் is frequent in the corpus; -த்தாழ never occurs
    assert lm.logp("அளித்தான்") > lm.logp("அளித்தாழ")


def test_extract_glyph_patch_shape():
    p = extract_glyph_patch(np.eye(20, dtype=np.float32), 0, 20)
    assert p.shape == (56, 56)


def test_enhance_accepts_uint8_and_bgr_photos():
    """``cv2.imread`` hands us uint8 0-255 BGR; enhance must not clip it to white."""
    rng = random.Random(21)
    c = render_canvas([sample_tamil_line(rng).text for _ in range(2)], "tamil",
                      cfg=DamageConfig(level=2), rng=rng)
    bgr = (np.clip(c.image, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    assert enhance(bgr).max() > 0.5, "uint8 BGR photo enhanced to a blank map"
    assert enhance(bgr.astype(np.float32)).max() > 0.5, "0-255 float photo blank"
    assert len(segment_lines(enhance(bgr))) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
