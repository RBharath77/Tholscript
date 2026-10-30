"""Per-glyph convolutional classifier.

The glyph CNN is the *classical* path of the pipeline: segment a character,
classify it.  The CRNN reads the whole line and the glyph CNN cross-checks each
predicted character on its own image patch, so the final confidence for every
output character is the agreement of two independent models.  The CNN also owns
an explicit ``<unk>`` class ("this patch is not a readable character": noise,
crack fragments, chipped blobs) which the report uses to flag doubtful spots.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphCNN(nn.Module):
    def __init__(self, num_classes: int, size: int = 56):
        super().__init__()
        self.size = size
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (size // 16) ** 2, 256), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(x), dim=-1)
