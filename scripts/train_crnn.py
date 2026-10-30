"""Train the CRNN + CTC line reader for one script, in two phases.

Phase 0  *dense frame supervision.*  The synthetic renderer knows every
         character's x-box, so the conv stack is first trained as a plain
         per-column classifier (column label = the cluster whose box covers
         it, blank elsewhere).  Dense supervision converges in a few hundred
         steps and teaches the features that CTC would otherwise spend
         thousands of steps discovering.
Phase 1  *CTC fine-tune.*  The whole network (conv warm-started) is trained
         end-to-end with CTC on damaged photo strips pushed through the same
         ``enhance`` the web app uses, with a short clean-line curriculum.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tholscript.config import ARTIFACT_DIR, LINE_HEIGHT, DamageConfig, TrainConfig
from tholscript.data.inventory import classes_for, clusterize
from tholscript.data.lexicon import sample_line
from tholscript.models.crnn import CRNN, ctc_loss, greedy_decode
from tholscript.vision.preprocess import enhance
from tholscript.vision.renderer import render_line


class LineDataset(Dataset):
    def __init__(self, script: str, n: int, seed: int,
                 level_range=(0, 3), word_range=(3, 9)):
        self.script = script
        self.n = n
        self.seed = seed
        self.level_range = level_range
        self.word_range = word_range
        self.classes = classes_for(script)
        self.idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        rng = random.Random(self.seed * 1000003 + i)
        line = sample_line(rng, self.script)
        if rng.random() < 0.7:
            k = rng.randint(*self.word_range)
            words = line.text.split()
            line.text = " ".join(words[:max(2, min(k, len(words)))])
        level = rng.randint(*self.level_range)
        s = render_line(line.text, script=self.script,
                        cfg=DamageConfig(level=level), rng=rng,
                        seed=rng.randint(0, 2 ** 31))
        ink = enhance(s.image, clahe=rng.random() < 0.4)   # real photos get CLAHE
        h, w = ink.shape
        scale = LINE_HEIGHT / h
        nw = min(1200, max(24, int(w * scale)))
        ink = cv2.resize(ink, (nw, LINE_HEIGHT), interpolation=cv2.INTER_AREA)

        clusters = clusterize(line.text, self.script)
        targets = [self.idx[c] for c in clusters]
        if not targets:
            targets = [1]

        # dense column labels in pooled coordinates (for phase 0)
        T = max(1, nw // CRNN.WIDTH_POOL)
        col = torch.zeros(T, dtype=torch.long)
        sx = nw / max(w, 1)
        for c in s.chars:
            lab = self.idx.get(c.text)
            if lab is None:
                continue
            t0 = int((c.x0 * sx) // CRNN.WIDTH_POOL)
            t1 = int((c.x1 * sx) // CRNN.WIDTH_POOL) + 1
            col[max(0, t0):min(T, t1)] = lab
        return (torch.from_numpy(ink).float().unsqueeze(0),
                torch.tensor(targets, dtype=torch.long),
                col, s.text)


def collate(batch):
    w = max(x.size(2) for x, _, _, _ in batch)
    xs = torch.zeros(len(batch), 1, LINE_HEIGHT, w)
    T = max(c.size(0) for _, _, c, _ in batch)
    cols = torch.full((len(batch), T), 0, dtype=torch.long)
    targets, texts = [], []
    for i, (x, t, c, txt) in enumerate(batch):
        xs[i, :, :, :x.size(2)] = x
        cols[i, :c.size(0)] = c
        targets.append(t.tolist())
        texts.append(txt)
    return xs, targets, cols, texts


def cer(pred_ids, gt_ids):
    import difflib
    if not gt_ids:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, gt_ids, pred_ids).ratio()


def main(script: str = "tamil", cfg: TrainConfig = None) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    classes = classes_for(script)
    print(f"[crnn:{script}] classes={len(classes)}")

    model = CRNN(len(classes), height=LINE_HEIGHT)

    # ------------------------------------------------------------------ #
    # phase 0: dense frame supervision on the conv stack + head
    # ------------------------------------------------------------------ #
    pre_steps = cfg.crnn_pretrain_steps
    ds0 = LineDataset(script, pre_steps * cfg.crnn_batch_size, cfg.seed,
                      level_range=(0, 1), word_range=(3, 6))
    workers = 0   # this box has 2 GB RAM; extra workers get OOM-killed
    dl0 = DataLoader(ds0, batch_size=cfg.crnn_batch_size, shuffle=True,
                     collate_fn=collate, num_workers=workers)
    frame_head = torch.nn.Linear(96 * (LINE_HEIGHT // 8), len(classes))
    opt = torch.optim.Adam(list(model.conv.parameters()) + list(frame_head.parameters()),
                           lr=3e-3)
    model.train()
    t0 = time.time()
    step = 0
    for x, _, cols, _ in dl0:
        if step >= pre_steps:
            break
        opt.zero_grad()
        logits = frame_head(model.features(x))           # (B, T, C)
        T = min(logits.size(1), cols.size(1))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :T].reshape(-1, len(classes)), cols[:, :T].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            with torch.no_grad():
                acc = (logits[:, :T].argmax(-1) == cols[:, :T]).float().mean()
            print(f"[crnn:{script}] pretrain {step:4d}/{pre_steps} "
                  f"loss={loss.item():.3f} col_acc={acc:.3f} ({time.time() - t0:.0f}s)")
    print(f"[crnn:{script}] pretrain done in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------ #
    # phase 1: CTC fine-tune with curriculum
    # ------------------------------------------------------------------ #
    val = LineDataset(script, cfg.crnn_val_samples, cfg.seed + 7)
    va = DataLoader(val, batch_size=8, shuffle=False, collate_fn=collate)
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0)

    best = 10.0
    history = []
    n_easy = max(1, cfg.crnn_epochs // 3)
    for ep in range(cfg.crnn_epochs):
        if ep < n_easy:
            level_range, word_range = (0, 1), (3, 6)
        else:
            level_range, word_range = (0, 3), (3, 9)
        train = LineDataset(script, cfg.crnn_samples_per_epoch, cfg.seed + ep,
                            level_range=level_range, word_range=word_range)
        tr = DataLoader(train, batch_size=cfg.crnn_batch_size, shuffle=True,
                        collate_fn=collate, num_workers=workers)
        model.train()
        tot, loss_sum = 0, 0.0
        for x, targets, _, _ in tr:
            opt.zero_grad()
            logits = model(x)
            loss = ctc_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += sum(len(t) for t in targets)
            loss_sum += loss.item() * sum(len(t) for t in targets)

        model.eval()
        errs, chars = 0, 0
        with torch.no_grad():
            for x, targets, _, _ in va:
                logits = model(x)
                for b in range(x.size(0)):
                    dec = greedy_decode(logits[:, b, :])
                    ids = [i for i, _ in dec if i < len(classes)]
                    errs += cer(ids, targets[b]) * len(targets[b])
                    chars += len(targets[b])
        vcer = errs / max(chars, 1)
        history.append(dict(epoch=ep, loss=loss_sum / max(tot, 1), val_cer=vcer))
        print(f"[crnn:{script}] ep {ep:2d} loss={loss_sum / max(tot, 1):.4f} "
              f"val_cer={vcer:.3f} ({time.time() - t0:.0f}s)")
        if vcer < best:
            best = vcer
            ckpt = ARTIFACT_DIR / f"crnn_{script}.pt"
            torch.save(dict(model=model.state_dict(), classes=classes,
                            script=script, height=LINE_HEIGHT), ckpt)
    print(f"[crnn:{script}] best val_cer={best:.3f} in {time.time() - t0:.0f}s")
    (ARTIFACT_DIR / f"crnn_classes_{script}.json").write_text(
        json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    return dict(script=script, history=history, best_cer=best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="tamil")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = TrainConfig()
    if args.epochs:
        cfg.crnn_epochs = args.epochs
    main(args.script, cfg)
