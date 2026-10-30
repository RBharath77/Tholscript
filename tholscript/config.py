"""Central configuration for the TholScript ancient-script translator.

Everything that a user would plausibly want to change lives here so that the
training scripts, the inference pipeline and the web app all agree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent          # .../tholscript
FONT_DIR = ROOT / "fonts"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
for _d in (ARTIFACT_DIR, DATA_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FONTS = {
    # script -> {style -> path}
    "tamil": {
        "sans": FONT_DIR / "NotoSansTamil-Regular.ttf",
        "sans_bold": FONT_DIR / "NotoSansTamil-Bold.ttf",
        "serif": FONT_DIR / "NotoSerifTamil-Regular.ttf",
    },
    "brahmi": {
        "sans": FONT_DIR / "NotoSansBrahmi-Regular.ttf",
    },
}

# --------------------------------------------------------------------------- #
# Synthetic inscription rendering
# --------------------------------------------------------------------------- #
IMAGE_HEIGHT = 88            # height (px) of the rendered "stone line" strip
MAX_LINE_WIDTH = 1500        # safety cap while composing a line
FONT_SIZE_RANGE = (54, 72)
CHAR_SPACING_RANGE = (2, 9)  # extra px between grapheme clusters

# --------------------------------------------------------------------------- #
# Model geometry
# --------------------------------------------------------------------------- #
GLYPH_SIZE = 56              # square patch fed to the glyph CNN
LINE_HEIGHT = 64             # height fed to the CRNN line reader
LINE_WIDTH = 960             # width fed to the CRNN line reader


@dataclass
class DamageConfig:
    """Probabilities / strengths of the degradation pipeline.

    ``level`` (0..3) scales every probability, so we can measure the
    accuracy-vs-damage curve that makes an OCR model for *ruined* stone useful.
    """
    level: int = 2
    p_erode: float = 0.55
    p_dilate: float = 0.15
    p_crack: float = 0.75
    p_chipping: float = 0.70
    p_lichen: float = 0.55
    p_stone_texture: float = 0.90
    p_lighting: float = 0.80
    p_blur: float = 0.70
    p_noise: float = 0.80
    p_perspective: float = 0.45
    p_partial_loss: float = 0.50   # a whole chunk of the line worn away
    max_level: int = 3

    def scale(self, name: str) -> float:
        """Probability of applying effect `name` at the configured level."""
        base = getattr(self, f"p_{name}")
        return min(1.0, base * (0.35 + 0.35 * self.level))


@dataclass
class TrainConfig:
    epochs: int = 9
    batch_size: int = 96
    lr: float = 3e-4
    weight_decay: float = 1e-5
    samples_per_epoch: int = 3400
    val_samples: int = 700
    seed: int = 20260910
    num_workers: int = 0
    ckpt: str = "glyph_cnn.pt"

    crnn_pretrain_steps: int = 250
    crnn_epochs: int = 10
    crnn_batch_size: int = 24
    crnn_lr: float = 3e-3
    crnn_samples_per_epoch: int = 600
    crnn_val_samples: int = 90
    crnn_ckpt: str = "crnn_line.pt"

    def to_dict(self) -> Dict:
        return asdict(self)


DEFAULT_DAMAGE = DamageConfig()
DEFAULT_TRAIN = TrainConfig()

THREADS = int(os.environ.get("THOLSCRIPT_THREADS", "0"))  # 0 => let torch decide


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware accelerators."""
    env_dev = os.environ.get("THOLSCRIPT_DEVICE")
    if env_dev:
        return env_dev
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = get_device()


def torch_threads() -> int:
    if THREADS > 0:
        return THREADS
    try:
        import torch
        return max(1, torch.get_num_threads())
    except Exception:
        return 1
