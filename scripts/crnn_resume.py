"""Resume CRNN CTC fine-tuning from the saved checkpoint.

``train_crnn.main`` always starts from scratch, which wastes the (expensive)
dense-frame pretrain when the CTC phase simply needs more epochs.  This script
loads ``artifacts/crnn_<script>.pt``, re-uses the already warm conv stack and
continues the CTC curriculum, overwriting the checkpoint whenever validation
CER improves.

    python -m tholscript.scripts.crnn_resume --script tamil --epochs 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tholscript.config import ARTIFACT_DIR, TrainConfig
from tholscript.models.crnn import CRNN, ctc_loss, greedy_decode
from tholscript.scripts.train_crnn import LineDataset, cer, collate


def main(script: str, epochs: int = 20, cfg: TrainConfig = None) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    ck = ARTIFACT_DIR / f"crnn_{script}.pt"
    d = torch.load(ck, map_location="cpu", weights_only=False)
    classes = d["classes"]
    model = CRNN(len(classes), height=d.get("height", 64))
    model.load_state_dict(d["model"])
    print(f"[resume:{script}] classes={len(classes)} ckpt={ck.name}", flush=True)

    val = LineDataset(script, cfg.crnn_val_samples, cfg.seed + 7)
    va = DataLoader(val, batch_size=8, shuffle=False, collate_fn=collate)
    opt = torch.optim.Adadelta(model.parameters(), lr=0.5)
    workers = 0   # this box has 2 GB RAM; extra workers get OOM-killed

    best = 10.0
    history = []
    n_easy = max(1, epochs // 3)
    t0 = time.time()
    for ep in range(epochs):
        if ep < n_easy:
            level_range, word_range = (0, 1), (3, 6)
        else:
            level_range, word_range = (0, 3), (3, 9)
        train = LineDataset(script, cfg.crnn_samples_per_epoch, cfg.seed + 500 + ep,
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
        print(f"[resume:{script}] ep {ep:2d} loss={loss_sum / max(tot, 1):.4f} "
              f"val_cer={vcer:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if vcer < best:
            best = vcer
            torch.save(dict(model=model.state_dict(), classes=classes,
                            script=script, height=d.get("height", 64)), ck)
    print(f"[resume:{script}] best val_cer={best:.3f} in {time.time() - t0:.0f}s",
          flush=True)
    return dict(script=script, history=history, best_cer=best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="tamil")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    main(args.script, args.epochs)
