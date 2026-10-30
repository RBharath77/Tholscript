"""The inference pipeline: photo -> enhanced ink -> lines -> characters -> text.

Stages and the model behind each:

1. ``enhance``      stone-photo -> ink map            (vision.preprocess)
2. ``deskew``       principal-axis rotation fix       (vision.preprocess)
3. ``segment_lines`` horizontal projection bands      (vision.preprocess)
4. ``read_line``    CRNN + CTC greedy decode          (models.crnn)
5. ``cross_check``  glyph CNN on each predicted char  (models.glyph_cnn)
6. ``correct``      lexicon + LM spell check          (translation.spellcheck)

The per-character confidence shown in the UI is the agreement of the two
independent readers (step 4 and step 5); characters the glyph CNN rejects are
flagged as doubtful and rendered as "·" candidates in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from tholscript.config import ARTIFACT_DIR, GLYPH_SIZE, LINE_HEIGHT, get_device
from tholscript.models.crnn import CRNN, greedy_decode, decode_with_spans, beam_search_decode
from tholscript.models.glyph_cnn import GlyphCNN
from tholscript.translation.spellcheck import CorrectedLine, SpellChecker
from tholscript.vision.preprocess import enhance, extract_glyph_patch, segment_lines, deskew


@dataclass
class CharReading:
    cluster: str
    x0: int
    x1: int
    crnn_conf: float
    glyph_conf: float
    glyph_agrees: bool
    fused_conf: float
    doubtful: bool
    glyph_top_cls: str = ""
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class LineReading:
    y0: int
    y1: int
    text: str                      # raw CRNN reading
    chars: List[CharReading] = field(default_factory=list)
    corrected: Optional[CorrectedLine] = None
    mean_conf: float = 0.0


@dataclass
class ReadingResult:
    script: str
    lines: List[LineReading]
    enhanced: np.ndarray
    line_boxes: List[Tuple[int, int, int, int]]
    full_text: str
    corrected_text: str


class Recognizer:
    def __init__(self, scripts=("tamil", "brahmi"), spell: Optional[SpellChecker] = None):
        self.device = torch.device(get_device())
        self.crnns, self.glyphs, self.classes = {}, {}, {}
        self.glyph_classes = {}
        for s in scripts:
            ck = ARTIFACT_DIR / f"crnn_{s}.pt"
            gk = ARTIFACT_DIR / f"glyph_cnn_{s}.pt"
            if ck.exists():
                try:
                    d = torch.load(ck, map_location=self.device, weights_only=False)
                    m = CRNN(len(d["classes"]), height=d.get("height", LINE_HEIGHT)).to(self.device)
                    m.load_state_dict(d["model"])
                    m.eval()
                    self.crnns[s] = m
                    self.classes[s] = d["classes"]
                except Exception as e:
                    print(f"[recognizer] skipping stale checkpoint {ck.name}: {e}")
            if gk.exists():
                try:
                    d = torch.load(gk, map_location=self.device, weights_only=False)
                    g = GlyphCNN(len(d["classes"])).to(self.device)
                    g.load_state_dict(d["model"])
                    g.eval()
                    self.glyphs[s] = g
                    self.glyph_classes[s] = d["classes"]
                except Exception as e:
                    print(f"[recognizer] skipping stale checkpoint {gk.name}: {e}")

    # ------------------------------------------------------------------ #
    def read_line(self, ink_line: np.ndarray, script: str,
                  beam_search: bool = False,
                  return_spans: bool = False):
        if script not in self.crnns:
            raise RuntimeError(
                f"no CRNN checkpoint for script {script!r} in artifacts/ — "
                f"run: python -m tholscript.scripts.cli train --script {script} --model crnn")
        model = self.crnns[script]
        h, w = ink_line.shape
        scale = LINE_HEIGHT / h
        nw = max(24, int(w * scale))
        x = cv2.resize(ink_line, (nw, LINE_HEIGHT), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(t)[:, 0, :]
        if beam_search:
            dec_items = beam_search_decode(logits)
        else:
            dec_items = decode_with_spans(logits)
        classes = self.classes[script]
        out = []
        spans = []
        for item in dec_items:
            i, conf, s_ratio, e_ratio = item
            if 0 < i < len(classes):
                out.append((classes[i], conf))
                spans.append((s_ratio, e_ratio))
        if not out:
            return ([], [], []) if return_spans else ([], [])
        clusters = [c for c, _ in out]
        confs = [c for _, c in out]
        if return_spans:
            return clusters, confs, spans
        return clusters, confs

    def cross_check(self, ink_line: np.ndarray, clusters: List[str],
                    confs: List[float], script: str,
                    spans: Optional[List[Tuple[float, float]]] = None) -> List[CharReading]:
        gmodel = self.glyphs.get(script)
        h, w = ink_line.shape
        readings: List[CharReading] = []
        n = max(1, len(clusters))
        classes = self.glyph_classes.get(script, self.classes.get(script, []))
        unk_i = classes.index("<unk>") if "<unk>" in classes else -1

        for i, (cluster, conf) in enumerate(zip(clusters, confs)):
            if spans and i < len(spans):
                s0, s1 = spans[i]
                x0 = max(0, int(w * s0))
                x1 = min(w, int(w * s1))
                if x1 - x0 < 12:
                    mid = (x0 + x1) // 2
                    x0 = max(0, mid - 14)
                    x1 = min(w, mid + 14)
            else:
                x0 = int(w * i / n)
                x1 = int(w * (i + 1) / n)

            g_conf, agrees, doubtful = 0.0, False, conf < 0.5
            top_cls = ""
            alts: List[Tuple[str, float]] = []

            if gmodel is not None and classes:
                patch = extract_glyph_patch(ink_line, x0, x1, GLYPH_SIZE)
                t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = gmodel.probabilities(t)[0].cpu()
                top_vals, top_idxs = probs.topk(min(4, len(classes)))
                top_p, top_i = top_vals[0].item(), top_idxs[0].item()
                g_conf = float(top_p)
                top_cls = classes[top_i] if top_i < len(classes) else ""
                agrees = (top_cls == cluster)
                doubtful = (not agrees and g_conf > 0.6) or conf < 0.35 or \
                    (unk_i >= 0 and top_i == unk_i and g_conf > 0.6)

                for val, idx in zip(top_vals, top_idxs):
                    c_idx = idx.item()
                    if c_idx < len(classes) and classes[c_idx] != "<unk>":
                        alts.append((classes[c_idx], round(float(val.item()), 3)))

            fused = 0.55 * conf + 0.45 * (g_conf if agrees else 0.5 * g_conf)
            readings.append(CharReading(
                cluster=cluster, x0=x0, x1=x1, crnn_conf=conf,
                glyph_conf=g_conf, glyph_agrees=agrees, fused_conf=fused,
                doubtful=doubtful, glyph_top_cls=top_cls, alternatives=alts
            ))
        return readings

    # ------------------------------------------------------------------ #
    def read_photo(self, photo: np.ndarray, script: str = "tamil",
                   spell: Optional[SpellChecker] = None,
                   line_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                   ) -> ReadingResult:
        ink = enhance(photo)
        ink, _ = deskew(ink)
        if line_boxes is None:
            bands = segment_lines(ink)
        else:
            bands = [(b[1], b[3]) for b in line_boxes]
        lines: List[LineReading] = []
        boxes: List[Tuple[int, int, int, int]] = []
        for (y0, y1) in bands:
            y0, y1 = int(y0), int(y1)
            strip = ink[y0:y1]
            if strip.sum() < 20:
                continue
            ys, xs = np.where(strip > 0.2)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            strip = strip[:, x0:x1]
            clusters, confs, spans = self.read_line(strip, script, return_spans=True)
            if not clusters:
                continue
            readings = self.cross_check(strip, clusters, confs, script, spans=spans)
            text = "".join(clusters)
            lr = LineReading(y0=y0, y1=y1, text=text, chars=readings,
                             mean_conf=float(np.mean([r.fused_conf for r in readings])))
            if spell is not None:
                lr.corrected = spell.correct(" ".join(_split_words(strip, clusters)))
            lines.append(lr)
            boxes.append((x0, y0, x1, y1))
        full = "\n".join(l.text for l in lines)
        corrected = "\n".join(
            (l.corrected.text if l.corrected else l.text) for l in lines)
        return ReadingResult(script=script, lines=lines, enhanced=ink,
                             line_boxes=boxes, full_text=full,
                             corrected_text=corrected)


def _split_words(strip: np.ndarray, clusters: List[str],
                 min_gap: int = 10) -> List[str]:
    """Word splitting from inter-word blank columns.

    Inter-word gaps on a rendered line are much wider than inter-cluster
    spacing, so blank column runs of ``min_gap`` or more mark word breaks.
    The break columns are then mapped onto the (uniformly placed) predicted
    clusters to regroup them into words for the spell-checker.
    """
    n = len(clusters)
    if n == 0:
        return []
    cols = (strip > 0.15).sum(axis=0)
    w = len(cols)
    breaks = []
    run = 0
    for x in range(w):
        if cols[x] == 0:
            run += 1
        else:
            if run >= min_gap and 0 < x - run // 2 < w:
                breaks.append(x - run // 2)
            run = 0
    words: List[str] = []
    cur: List[str] = []
    b = 0
    for i, c in enumerate(clusters):
        centre = w * (i + 0.5) / n
        while b < len(breaks) and centre > breaks[b]:
            if cur:
                words.append("".join(cur))
                cur = []
            b += 1
        cur.append(c)
    if cur:
        words.append("".join(cur))
    return words


def detect_script(photo: np.ndarray, recognizer: "Recognizer") -> str:
    """Pick the script whose CRNN is more confident on the first line."""
    ink = enhance(photo)
    bands = segment_lines(ink)
    if not bands:
        return "tamil"
    y0, y1 = bands[0]
    scores = {}
    for s in recognizer.crnns:
        clusters, confs = recognizer.read_line(ink[y0:y1], s)
        scores[s] = float(np.mean(confs)) if confs else 0.0
    return max(scores, key=scores.get)
