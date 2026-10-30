"""Recognition-error correction for the transliterated output.

The recognizer reads *damaged* characters, so its raw words sometimes drift by
one grapheme from a real inscription word.  ``correct`` snaps each unknown word
to the lexicon candidate with the best (edit distance, LM prior) score and
reports every change so the UI can show what was fixed and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tholscript.models.lm import CharLM


@dataclass
class Correction:
    original: str
    corrected: str
    distance: int
    reason: str


@dataclass
class CorrectedLine:
    text: str
    words: List[str]
    changes: List[Correction] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


def _edit_distance(a: str, b: str, cap: int = 4) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class SpellChecker:
    def __init__(self, lexicon: List[str], lm: Optional[CharLM] = None):
        self.lexicon = sorted(set(lexicon))
        self.lex_set = set(self.lexicon)
        self.lm = lm

    def _candidate(self, word: str) -> Optional[Tuple[str, int]]:
        best: Optional[Tuple[str, int]] = None
        cap = max(1, len(word) // 3)
        for cand in self.lexicon:
            if abs(len(cand) - len(word)) > cap + 1:
                continue
            d = _edit_distance(word, cand, cap)
            if d > cap:
                continue
            score = d
            if self.lm is not None:
                score += 0.15 * min(self.lm.perplexity(cand), 50.0) / 50.0
            if best is None or score < best[1]:
                best = (cand, d)
        return best

    def correct(self, text: str) -> CorrectedLine:
        words = text.split()
        out: List[str] = []
        changes: List[Correction] = []
        unknown: List[str] = []
        for w in words:
            if w in self.lex_set or not w or w.isdigit():
                out.append(w)
                continue
            hit = self._candidate(w)
            if hit is None:
                out.append(w)
                unknown.append(w)
                continue
            cand, d = hit
            out.append(cand)
            changes.append(Correction(w, cand, d,
                                      f"edit distance {d} to lexicon word"))
        return CorrectedLine(text=" ".join(out), words=out, changes=changes,
                             unknown=unknown)


if __name__ == "__main__":
    from tholscript.data.lexicon import ALL_WORDS, build_word_lexicon
    from tholscript.models.lm import build_lm
    import random

    rng = random.Random(1)
    lm = build_lm([], ALL_WORDS)
    sc = SpellChecker(build_word_lexicon(), lm)
    for w in ["கொடஇ", "மதுரய்", "கோயில", "நல்லி", "ஊர", "அழகந"]:
        r = sc.correct(w)
        print(f"{w:8s} -> {r.text:8s} changes={[(c.original, c.corrected) for c in r.changes]}")
