"""CRNN + CTC line reader.

A small convolutional stack maps the enhanced line strip to a sequence of
column features; a bidirectional LSTM reads them left-to-right and right-to-left
and a linear head emits one logit vector per column.  ``CTCLoss`` trains the
whole thing against the *sequence of grapheme clusters* of the line, so no
character segmentation is needed at all -- which is exactly what a ruined
inscription demands, because its characters merge, chip and fade.

Architecture notes (each one measured the hard way, see docs/DESIGN.md):

* BatchNorm after every conv -- without it the stroke signal attenuates to
  noise and CTC collapses to all-blank;
* width pooled 8x -- ~6-10 CTC frames per character; at 40 frames/char the
  blank posterior dominates and gradients to real characters vanish;
* one LSTM layer, hidden 96 -- thousands of CPU steps must be affordable;
* the conv stack is normally warm-started with *dense frame supervision*
  (per-column classification against the known character boxes of the
  synthetic renders) before CTC fine-tuning; that turns a several-thousand-
  step CTC problem into a few-hundred-step one.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    WIDTH_POOL = 8

    def __init__(self, num_classes: int, height: int = 64, hidden: int = 96):
        super().__init__()
        self.height = height
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                      # H/2  W/2
            nn.Conv2d(24, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                      # H/4  W/4
            nn.Conv2d(48, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, (2, 2)),                 # H/8  W/8
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(True),
            nn.Dropout2d(0.15),
        )
        feat_h = height // 8
        self.rnn = nn.LSTM(96 * feat_h, hidden, num_layers=1, bidirectional=True,
                           batch_first=True)
        self.head = nn.Linear(hidden * 2, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 96*H/8) column features with T = W / WIDTH_POOL."""
        y = self.conv(x)
        b, c, h, w = y.shape
        return y.permute(0, 3, 1, 2).reshape(b, w, c * h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) with H == self.height.  Returns (T, B, C) logits."""
        y = self.features(x)
        y, _ = self.rnn(y)
        return self.head(y).permute(1, 0, 2)          # (T, B, C)


# --------------------------------------------------------------------------- #
# CTC helpers
# --------------------------------------------------------------------------- #
def greedy_decode(logits: torch.Tensor) -> List[Tuple[int, float]]:
    """(T, C) logits -> list of (class_index, mean softmax confidence)."""
    probs = F.softmax(logits, dim=-1)
    best = probs.max(dim=-1)
    idx = best.indices.tolist()
    conf = best.values.tolist()
    out: List[Tuple[int, float]] = []
    spans: List[List[float]] = []
    prev = None
    for t, (i, c) in enumerate(zip(idx, conf)):
        if i != 0 and i != prev:
            out.append((i, c))
            spans.append([c])
        elif i != 0 and i == prev and spans:
            spans[-1].append(c)
        prev = i
    return [(i, sum(s) / len(s)) for (i, _), s in zip(out, spans)]


def decode_with_spans(logits: torch.Tensor) -> List[Tuple[int, float, float, float]]:
    """(T, C) logits -> list of (class_index, mean_conf, start_ratio, end_ratio).

    start_ratio and end_ratio are in [0, 1] relative to line width, indicating
    where the character activation occurred along the image strip.
    """
    probs = F.softmax(logits, dim=-1)
    best = probs.max(dim=-1)
    idx = best.indices.tolist()
    conf = best.values.tolist()
    T = max(1, len(idx))
    out: List[int] = []
    spans: List[List[float]] = []
    times: List[List[int]] = []
    prev = None
    for t, (i, c) in enumerate(zip(idx, conf)):
        if i != 0 and i != prev:
            out.append(i)
            spans.append([c])
            times.append([t, t + 1])
        elif i != 0 and i == prev and spans:
            spans[-1].append(c)
            times[-1][1] = t + 1
        prev = i
    result = []
    for cls, s, (t0, t1) in zip(out, spans, times):
        result.append((cls, float(sum(s) / len(s)), t0 / T, t1 / T))
    return result


def beam_search_decode(logits: torch.Tensor, beam_width: int = 8) -> List[Tuple[int, float, float, float]]:
    """Prefix beam search for CTC logits -> (class_index, conf, start_ratio, end_ratio)."""
    import numpy as np

    T, C = logits.shape
    log_probs = F.log_softmax(logits, dim=-1).detach().cpu().numpy()
    NEG_INF = -1e9
    beams = {(): (0.0, NEG_INF, [])}

    for t in range(T):
        next_beams = {}
        top_k = min(C, max(8, beam_width * 2))
        top_indices = np.argsort(log_probs[t])[::-1][:top_k]

        for prefix, (p_b, p_nb, spans) in beams.items():
            p_total = np.logaddexp(p_b, p_nb) if (p_b > NEG_INF or p_nb > NEG_INF) else NEG_INF

            # blank
            lp_b = log_probs[t, 0]
            cur_b, cur_nb, cur_sp = next_beams.get(prefix, (NEG_INF, NEG_INF, spans))
            next_beams[prefix] = (np.logaddexp(cur_b, p_total + lp_b), cur_nb, cur_sp)

            for c in top_indices:
                if c == 0:
                    continue
                lp_c = log_probs[t, c]
                prob_c = float(np.exp(lp_c))

                if prefix and prefix[-1] == c:
                    new_p = prefix + (int(c),)
                    new_sp = spans + [(int(c), prob_c, t, t + 1)]
                    cur_b2, cur_nb2, _ = next_beams.get(new_p, (NEG_INF, NEG_INF, new_sp))
                    next_beams[new_p] = (cur_b2, np.logaddexp(cur_nb2, p_b + lp_c), new_sp)

                    cur_b1, cur_nb1, cur_sp1 = next_beams.get(prefix, (NEG_INF, NEG_INF, spans))
                    if spans:
                        up_sp = spans[:-1] + [(spans[-1][0], (spans[-1][1] + prob_c) / 2, spans[-1][2], t + 1)]
                    else:
                        up_sp = [(int(c), prob_c, t, t + 1)]
                    next_beams[prefix] = (cur_b1, np.logaddexp(cur_nb1, p_nb + lp_c), up_sp)
                else:
                    new_p = prefix + (int(c),)
                    new_sp = spans + [(int(c), prob_c, t, t + 1)]
                    cur_b2, cur_nb2, _ = next_beams.get(new_p, (NEG_INF, NEG_INF, new_sp))
                    next_beams[new_p] = (cur_b2, np.logaddexp(cur_nb2, p_total + lp_c), new_sp)

        sorted_b = sorted(
            next_beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True
        )[:beam_width]
        beams = dict(sorted_b)

    best_prefix, (best_b, best_nb, best_spans) = max(
        beams.items(),
        key=lambda x: np.logaddexp(x[1][0], x[1][1])
    )
    return [(int(c), float(conf), t0 / T, t1 / T) for c, conf, t0, t1 in best_spans]


def ctc_loss(logits: torch.Tensor, targets: List[List[int]],
             input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """logits: (T, B, C)."""
    T, B, _ = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    if input_lengths is None:
        input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.tensor([i for t in targets for i in t], dtype=torch.long)
    return F.ctc_loss(log_probs, flat, input_lengths, target_lengths, blank=0,
                      zero_infinity=True)
