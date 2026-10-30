"""Glue between the trained models and the web UI / CLI.

One ``TranslateService`` owns the recognizer, the spell-checker and the demo
generator so the Flask app and the command line tool behave identically.
"""
from __future__ import annotations

import base64
import io
import random
from typing import Dict, List, Optional

import cv2
import numpy as np

from tholscript.config import DATA_DIR, DamageConfig
from tholscript.data.lexicon import (
    ALL_WORDS, build_word_lexicon, gloss_sentence, sample_brahmi_line,
    sample_tamil_line,
)
from tholscript.models.lm import build_lm
from tholscript.recognition.pipeline import Recognizer, detect_script
from tholscript.translation.spellcheck import SpellChecker
from tholscript.translation.translate import summarise
from tholscript.translation.voice import make_voice_payload
from tholscript.translation.epidoc import to_epidoc
from tholscript.vision.renderer import render_canvas, save_png


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (np.clip(img, 0, 1) * 255).astype(np.uint8))
    return base64.b64encode(buf.tobytes()).decode("ascii")


def from_b64(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


class TranslateService:
    def __init__(self):
        self.rec = Recognizer()
        self.lm = build_lm([], ALL_WORDS)
        self.spell = SpellChecker(build_word_lexicon(), self.lm)
        self._demo_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    def demo(self, key: str, seed: Optional[int] = None) -> Dict:
        if key in self._demo_cache and seed is None:
            return self._demo_cache[key]
        if key == "real_photo":
            path = DATA_DIR / "samples" / "real_inscription.jpg"
            bgr = cv2.imread(str(path))
            if bgr is None:
                return self.demo("tamil_medium", seed)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return dict(
                key=key, script="tamil", image_b64=_b64(img), gt_lines=None,
                damage=["real photograph — no ground truth, no damage dial"],
                width=img.shape[1], height=img.shape[0],
            )
        rng = random.Random(seed if seed is not None else random.randint(0, 10 ** 9))
        cfg_by_key = {
            "tamil_medium": ("tamil", DamageConfig(level=2), 4),
            "tamil_heavy": ("tamil", DamageConfig(level=3), 4),
            "tamil_light": ("tamil", DamageConfig(level=1), 4),
            "brahmi": ("brahmi", DamageConfig(level=2), 3),
        }
        script, cfg, n = cfg_by_key.get(key, cfg_by_key["tamil_medium"])
        texts = [sample_tamil_line(rng).text if script == "tamil"
                 else sample_brahmi_line(rng).text for _ in range(n)]
        canvas = render_canvas(texts, script=script, cfg=cfg, rng=rng)
        payload = dict(
            key=key, script=script,
            image_b64=_b64(canvas.image),
            gt_lines=[t for t in texts],
            damage=[s.trace.describe() for s in canvas.lines],
            width=canvas.image.shape[1], height=canvas.image.shape[0],
        )
        if seed is None:
            self._demo_cache[key] = payload
        return payload

    # ------------------------------------------------------------------ #
    def translate_image(self, img: np.ndarray, script: str = "auto") -> Dict:
        if script == "auto":
            script = detect_script(img, self.rec)
        result = self.rec.read_photo(img, script=script, spell=self.spell)
        out_lines = []
        for lr in result.lines:
            words = lr.corrected.words if lr.corrected else lr.text.split()
            summ = summarise(lr.text, script,
                             " ".join(words))["lines"][0] if words else None
            out_lines.append(dict(
                y0=lr.y0, y1=lr.y1,
                raw=lr.text,
                chars=[dict(cluster=c.cluster, x0=c.x0, x1=c.x1,
                            crnn_conf=round(c.crnn_conf, 3),
                            glyph_conf=round(c.glyph_conf, 3),
                            agrees=c.glyph_agrees,
                            fused=round(c.fused_conf, 3),
                            doubtful=c.doubtful,
                            glyph_top_cls=getattr(c, "glyph_top_cls", ""),
                            alternatives=getattr(c, "alternatives", [])) for c in lr.chars],
                corrected=" ".join(words) if lr.corrected else lr.text,
                changes=[dict(from_=c.original, to=c.corrected, distance=c.distance,
                              reason=c.reason)
                         for c in (lr.corrected.changes if lr.corrected else [])],
                unknown=(lr.corrected.unknown if lr.corrected else []),
                modern=summ["tamil"] if summ else "",
                roman=summ["roman"] if summ else [],
                gloss=summ["gloss_sentence"] if summ else "",
                mean_conf=round(lr.mean_conf, 3),
            ))
        full = summarise(result.full_text, script, result.corrected_text)
        reading_dict = dict(
            script=script,
            enhanced_b64=_b64(result.enhanced),
            line_boxes=result.line_boxes,
            lines=out_lines,
            full_text=result.full_text,
            corrected_text=result.corrected_text,
            modern=full["full_modern"],
            gloss=full["full_gloss"],
        )
        reading_dict["voice"] = make_voice_payload(reading_dict)
        reading_dict["epidoc_xml"] = to_epidoc(reading_dict)
        return reading_dict


if __name__ == "__main__":
    svc = TranslateService()
    d = svc.demo("tamil_medium")
    img = from_b64(d["image_b64"])
    r = svc.translate_image(img, script=d["script"])
    print("GT  :", d["gt_lines"])
    print("READ:", [l["raw"] for l in r["lines"]])
    print("CORR:", [l["corrected"] for l in r["lines"]])
