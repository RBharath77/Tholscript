"""Tiny character n-gram language model over Tamil grapheme clusters.

Trained on the synthetic inscription corpus (plus the lexicon itself).  Its job
is *not* generation -- it is a spelling prior: when the recognizer produces a
word that is not in the inscription lexicon, ``spellcheck`` proposes candidates
and the LM ranks them ("கொடஇ" is far less probable than "கொடை").
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

BOS, EOS = "<s>", "</s>"


class CharLM:
    def __init__(self, order: int = 3):
        self.order = order
        self.counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.vocab: set = set()

    def fit(self, words: Iterable[str]) -> "CharLM":
        for w in words:
            chars = [BOS] * (self.order - 1) + list(w) + [EOS]
            for i in range(self.order - 1, len(chars)):
                ctx = tuple(chars[i - self.order + 1:i])
                self.counts[ctx][chars[i]] += 1
                self.totals[ctx] += 1
                self.vocab.add(chars[i])
        return self

    def logp(self, word: str) -> float:
        chars = [BOS] * (self.order - 1) + list(word) + [EOS]
        lp = 0.0
        V = max(1, len(self.vocab))
        for i in range(self.order - 1, len(chars)):
            ctx = tuple(chars[i - self.order + 1:i])
            c = self.counts[ctx].get(chars[i], 0)
            t = self.totals.get(ctx, 0)
            lp += math.log((c + 0.5) / (t + 0.5 * V))     # add-half smoothing
        return lp

    def perplexity(self, word: str) -> float:
        n = max(1, len(word))
        return math.exp(-self.logp(word) / n)


def build_lm(texts: Iterable[str], lexicon: Iterable[str], seed: int = 5) -> CharLM:
    rng = random.Random(seed)
    lm = CharLM()
    words = list(lexicon)
    lm.fit(words)
    for t in texts:
        lm.fit(t.split())
    return lm


if __name__ == "__main__":
    from tholscript.data.lexicon import ALL_WORDS, sample_tamil_line

    rng = random.Random(0)
    lm = build_lm((sample_tamil_line(rng).text for _ in range(400)), ALL_WORDS)
    for w in ["கொடை", "கொடஇ", "மதுரை", "மதுரஐ", "கோயில்"]:
        print(f"{w:8s} logp={lm.logp(w):8.2f} ppl={lm.perplexity(w):6.2f}")
