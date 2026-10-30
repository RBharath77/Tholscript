"""Accuracy vs damage evaluation.

The whole point of a reader for *damaged* inscriptions is knowing how much
damage it tolerates, so every metric is reported per damage level 0..3:

* **CER**   character (grapheme cluster) error rate of the CRNN line reader
* **WER**   word error rate after the word splitter
* **EXACT** fraction of lines read perfectly
* **glyph** top-1 / top-3 accuracy of the isolated-glyph CNN on patches
            degraded at the same levels

Results go to ``reports/metrics.json`` and two charts.
"""
from __future__ import annotations

import difflib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tholscript.config import ARTIFACT_DIR, REPORT_DIR, DamageConfig, GLYPH_SIZE
from tholscript.data.inventory import clusterize, glyph_classes_for
from tholscript.data.lexicon import sample_line
from tholscript.models.glyph_cnn import GlyphCNN
from tholscript.recognition.pipeline import Recognizer, _split_words
from tholscript.vision.damage import degrade
from tholscript.vision.preprocess import enhance
from tholscript.vision.renderer import glyph_patch, render_line


def _cer(a: List[str], b: List[str]) -> float:
    if not a:
        return 0.0 if not b else 1.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def evaluate_crnn(rec: Recognizer, script: str, n_lines: int = 120,
                  seed: int = 99) -> Dict:
    rows = []
    for level in range(4):
        rng = random.Random(seed * 10 + level)
        cers, wers, exact = [], [], 0
        for _ in range(n_lines):
            line = sample_line(rng, script)
            s = render_line(line.text, script=script, cfg=DamageConfig(level=level),
                            rng=rng, seed=rng.randint(0, 2 ** 31))
            ink = enhance(s.image)
            ys, xs = np.where(ink > 0.2)
            if len(xs) == 0:
                cers.append(1.0)
                wers.append(1.0)
                continue
            strip = ink[:, xs.min():xs.max()]
            pred, confs = rec.read_line(strip, script)
            gt = clusterize(line.text, script)
            cers.append(_cer(gt, pred))
            gt_words = [clusterize(w, script) for w in line.text.split()]
            pred_words = []
            for wstr in _split_words(strip, pred):
                pred_words.append(clusterize(wstr, script))
            # word-level: compare joined strings
            g = ["".join(w) for w in gt_words]
            p = ["".join(w) for w in pred_words]
            wers.append(_cer(g, p))
            if "".join(pred) == "".join(gt):
                exact += 1
        rows.append(dict(level=level,
                         cer=round(float(np.mean(cers)), 3),
                         wer=round(float(np.mean(wers)), 3),
                         exact=round(exact / n_lines, 3)))
        print(f"[eval:{script}] level {level}: {rows[-1]}")
    return dict(script=script, lines=rows)


def evaluate_glyph(script: str, n: int = 250, seed: int = 7) -> Dict:
    ck = ARTIFACT_DIR / f"glyph_cnn_{script}.pt"
    d = torch.load(ck, map_location="cpu", weights_only=False)
    model = GlyphCNN(len(d["classes"]))
    model.load_state_dict(d["model"])
    model.eval()
    classes = d["classes"]
    idx = {c: i for i, c in enumerate(classes)}
    from tholscript.data.inventory import TAMIL_INVENTORY, BRAHMI_INVENTORY
    inv = TAMIL_INVENTORY if script == "tamil" else BRAHMI_INVENTORY
    rows = []
    for level in range(4):
        rng = random.Random(seed + level)
        top1 = top3 = 0
        for _ in range(n):
            cluster = rng.choice(inv)
            patch = glyph_patch(cluster, script=script, rng=rng)
            if patch is None:
                continue
            if level:
                patch, _ = degrade(patch, DamageConfig(level=level),
                                   seed=rng.randint(0, 2 ** 31))
            t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probs = model.probabilities(t)[0]
            order = probs.argsort(descending=True)[:3].tolist()
            lab = idx[cluster]
            if order[0] == lab:
                top1 += 1
            if lab in order:
                top3 += 1
        rows.append(dict(level=level, top1=round(top1 / n, 3), top3=round(top3 / n, 3)))
        print(f"[eval-glyph:{script}] level {level}: {rows[-1]}")
    return dict(script=script, glyph=rows)


def run_all(n_lines: int = 100, n_glyph: int = 200) -> Dict:
    t0 = time.time()
    rec = Recognizer()
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for script in ("tamil", "brahmi"):
        if script not in rec.crnns:
            print(f"[eval] no CRNN checkpoint for {script}, skipping")
            continue
        out[script] = evaluate_crnn(rec, script, n_lines=n_lines)
        out[script]["glyph"] = evaluate_glyph(script, n=n_glyph)["glyph"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"[eval] done in {time.time() - t0:.0f}s -> reports/metrics.json")
    return out


def make_charts(metrics: Dict) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in ("tamil", "brahmi"):
        if s not in metrics or "lines" not in metrics[s]:
            continue
        lv = [r["level"] for r in metrics[s]["lines"]]
        ax[0].plot(lv, [r["cer"] for r in metrics[s]["lines"]], marker="o", label=f"CER {s}")
        ax[0].plot(lv, [r["wer"] for r in metrics[s]["lines"]], marker="s", ls="--",
                   label=f"WER {s}")
        if "glyph" in metrics[s]:
            ax[1].plot(lv, [r["top1"] for r in metrics[s]["glyph"]], marker="o",
                       label=f"top1 {s}")
            ax[1].plot(lv, [r["top3"] for r in metrics[s]["glyph"]], marker="s", ls="--",
                       label=f"top3 {s}")
    ax[0].set_title("Line reader error vs damage level")
    ax[0].set_xlabel("damage level")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].set_title("Glyph CNN accuracy vs damage level")
    ax[1].set_xlabel("damage level")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    p = REPORT_DIR / "accuracy_vs_damage.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    paths.append(p)
    return paths


if __name__ == "__main__":
    m = run_all()
    make_charts(m)
