"""Command line interface.

    python -m tholscript.scripts.cli demo --key tamil_medium --show
    python -m tholscript.scripts.cli translate path/to/photo.png --script auto
    python -m tholscript.scripts.cli train --script tamil --model crnn --epochs 4
    python -m tholscript.scripts.cli evaluate
    python -m tholscript.scripts.cli serve --port 8080
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from tholscript.config import REPORT_DIR, ARTIFACT_DIR


def _img(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read image {path}")
    return g.astype(np.float32) / 255.0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tholscript")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("translate")
    p.add_argument("image")
    p.add_argument("--script", default="auto")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("demo")
    p.add_argument("--key", default="tamil_medium",
                   choices=["tamil_light", "tamil_medium", "tamil_heavy", "brahmi"])
    p.add_argument("--out", default="artifacts/demo.png")
    p.add_argument("--translate", action="store_true")

    p = sub.add_parser("train")
    p.add_argument("--script", default="tamil", choices=["tamil", "brahmi"])
    p.add_argument("--model", default="crnn", choices=["crnn", "glyph"])
    p.add_argument("--epochs", type=int, default=None)

    p = sub.add_parser("evaluate")
    p.add_argument("--lines", type=int, default=60)
    p.add_argument("--glyphs", type=int, default=120)

    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8080)

    p = sub.add_parser("train-all")
    p.add_argument("--quick", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "translate":
        from tholscript.app.service import TranslateService
        svc = TranslateService()
        r = svc.translate_image(_img(args.image), script=args.script)
        if args.json:
            r.pop("enhanced_b64", None)
            print(json.dumps(r, ensure_ascii=False, indent=2))
        else:
            print(f"script: {r['script']}\n")
            for i, l in enumerate(r["lines"], 1):
                print(f"line {i} [{(l['mean_conf'] * 100):.0f}%] {l['raw']}")
                if l["corrected"] != l["raw"]:
                    print(f"        corrected: {l['corrected']}")
                if l["modern"]:
                    print(f"        modern   : {l['modern']}")
                if l["roman"]:
                    print(f"        roman    : {' '.join(l['roman'])}")
                if l["gloss"]:
                    print(f"        gloss    : {l['gloss']}")
            print("\nfull modern text:", r["modern"])
            print("gloss           :", r["gloss"])

    elif args.cmd == "demo":
        from tholscript.app.service import TranslateService, from_b64
        svc = TranslateService()
        d = svc.demo(args.key)
        img = from_b64(d["image_b64"])
        cv2.imwrite(args.out, (np.clip(img, 0, 1) * 255).astype(np.uint8))
        print(f"wrote {args.out}")
        print("ground truth:")
        for t in d["gt_lines"]:
            print("  ", t)
        if args.translate:
            r = svc.translate_image(img, script=d["script"])
            for l in r["lines"]:
                print("  read:", l["corrected"])

    elif args.cmd == "train":
        from tholscript.config import TrainConfig
        from tholscript.scripts.train_crnn import main as cmain
        from tholscript.scripts.train_glyph import main as gmain
        cfg = TrainConfig()
        if args.epochs:
            cfg.epochs = cfg.crnn_epochs = args.epochs
        (cmain if args.model == "crnn" else gmain)(args.script, cfg)

    elif args.cmd == "evaluate":
        from tholscript.evaluation.evaluator import make_charts, run_all
        m = run_all(n_lines=args.lines, n_glyph=args.glyphs)
        make_charts(m)
        print(json.dumps(m, ensure_ascii=False)[:2000])

    elif args.cmd == "serve":
        from tholscript.app.main import app, service
        service()
        app.run(host="0.0.0.0", port=args.port, threaded=True)

    elif args.cmd == "train-all":
        from tholscript.scripts.train_all import run
        run(args.quick)


if __name__ == "__main__":
    main()
