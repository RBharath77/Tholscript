"""Train the per-glyph CNN for one script.

The dataset is generated on the fly: every batch renders fresh grapheme
clusters with fresh pose jitter, stroke width and damage, plus ~12% negative
patches cut out of the *stone* of degraded line renders -- that is exactly the
noise distribution the classifier will meet inside a ruined inscription.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tholscript.config import ARTIFACT_DIR, DamageConfig, TrainConfig
from tholscript.data.inventory import glyph_classes_for
from tholscript.data.inventory import TAMIL_INVENTORY, BRAHMI_INVENTORY
from tholscript.models.glyph_cnn import GlyphCNN
from tholscript.vision.damage import degrade
from tholscript.vision.renderer import glyph_patch, render_line


def inventory_for(script: str):
    return TAMIL_INVENTORY if script == "tamil" else BRAHMI_INVENTORY


class GlyphDataset(Dataset):
    def __init__(self, script: str, n: int, seed: int, level_range=(0, 3)):
        self.script = script
        self.n = n
        self.seed = seed
        self.classes = glyph_classes_for(script)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.inv = inventory_for(script)
        self.level_range = level_range

    def __len__(self):
        return self.n

    def _noise_patch(self, rng: random.Random) -> np.ndarray:
        line = render_line(" ".join(rng.choice(self.inv) for _ in range(6)),
                           script=self.script, cfg=DamageConfig(level=3), rng=rng)
        img = line.ink
        h, w = img.shape
        x = rng.randint(0, max(0, w - 70))
        patch = img[:, x:x + 64]
        ys, xs = np.where(patch > 0.25)
        # a noise patch should contain little/no coherent glyph: keep anyway,
        # the <unk> class is allowed to see glyph fragments.
        from tholscript.vision.preprocess import extract_glyph_patch
        return extract_glyph_patch(img, x, min(w, x + rng.randint(20, 60)))

    def __getitem__(self, i: int):
        rng = random.Random(self.seed * 1000003 + i)
        if rng.random() < 0.12:
            patch = self._noise_patch(rng)
            label = self.cls_to_idx["<unk>"]
        else:
            cluster = rng.choice(self.inv)
            patch = glyph_patch(cluster, script=self.script, rng=rng,
                                stroke=rng.randint(0, 2))
            if patch is None:
                patch = np.zeros((56, 56), np.float32)
                label = self.cls_to_idx["<unk>"]
            else:
                label = self.cls_to_idx[cluster]
                lvl = rng.randint(*self.level_range)
                if lvl > 0:
                    patch, _ = degrade(patch, DamageConfig(level=lvl),
                                       seed=rng.randint(0, 2 ** 31))
        patch = patch + rng.gauss(0, 0.02)
        t = torch.from_numpy(np.clip(patch, 0, 1)).float().unsqueeze(0)
        return t, torch.tensor(label, dtype=torch.long)


def main(script: str = "tamil", cfg: TrainConfig = None) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")
    classes = glyph_classes_for(script)
    print(f"[glyph:{script}] classes={len(classes)}")

    train = GlyphDataset(script, cfg.samples_per_epoch, cfg.seed)
    val = GlyphDataset(script, cfg.val_samples, cfg.seed + 1, level_range=(1, 2))
    tr = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    va = DataLoader(val, batch_size=256, shuffle=False)

    model = GlyphCNN(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot, hit, loss_sum = 0, 0, 0.0
        for x, y in tr:
            opt.zero_grad()
            logits = model(x.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += y.size(0)
            hit += (logits.argmax(-1) == y.to(device)).sum().item()
            loss_sum += loss.item() * y.size(0)
        sched.step()

        model.eval()
        vt, vh = 0, 0
        with torch.no_grad():
            for x, y in va:
                vt += y.size(0)
                vh += (model(x).argmax(-1) == y).sum().item()
        acc, vacc = hit / tot, vh / vt
        history.append(dict(epoch=ep, loss=loss_sum / tot, acc=acc, val_acc=vacc))
        print(f"[glyph:{script}] ep {ep:2d} loss={loss_sum / tot:.4f} "
              f"acc={acc:.3f} val_acc={vacc:.3f} ({time.time() - t0:.0f}s)")

    ckpt = ARTIFACT_DIR / f"glyph_cnn_{script}.pt"
    torch.save(dict(model=model.state_dict(), classes=classes, script=script), ckpt)
    (ARTIFACT_DIR / f"glyph_classes_{script}.json").write_text(
        json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    print(f"[glyph:{script}] saved {ckpt} in {time.time() - t0:.0f}s")
    return dict(script=script, history=history, ckpt=str(ckpt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="tamil")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    main(args.script, cfg)
