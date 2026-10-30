"""One command that trains everything, evaluates it and draws the charts.

    python -m tholscript.scripts.train_all            # full run (about 40-60 min CPU)
    python -m tholscript.scripts.train_all --quick    # smoke-scale run

Writes artifacts/*.pt, reports/metrics.json and reports/*.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tholscript.config import REPORT_DIR, TrainConfig
from tholscript.scripts.train_crnn import main as train_crnn
from tholscript.scripts.train_glyph import main as train_glyph


def run(quick: bool = False):
    t0 = time.time()
    cfg = TrainConfig()
    if quick:
        cfg.epochs, cfg.samples_per_epoch, cfg.val_samples = 2, 400, 120
        cfg.crnn_epochs, cfg.crnn_samples_per_epoch, cfg.crnn_val_samples = 2, 150, 40
        cfg.crnn_batch_size = 8
    hist = {}
    for script in ("tamil", "brahmi"):
        hist[f"glyph_{script}"] = train_glyph(script, cfg)
    for script in ("tamil", "brahmi"):
        hist[f"crnn_{script}"] = train_crnn(script, cfg)

    from tholscript.evaluation.evaluator import make_charts, run_all
    n_lines, n_glyph = (20, 50) if quick else (100, 200)
    metrics = run_all(n_lines=n_lines, n_glyph=n_glyph)
    paths = make_charts(metrics)
    print(f"[train_all] finished in {(time.time() - t0) / 60:.1f} min")
    print("[train_all] charts:", [str(p) for p in paths])
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
