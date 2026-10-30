"""Build the distributable zip of the whole project.

    python scripts/package.py          ->  ../tholscript.zip

Includes code, fonts, trained weights, reports, docs, tests and the VS Code
config.  Excludes caches, scratch images and logs.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "tholscript.zip"

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".venv", "node_modules", ".git"}
EXCLUDE_FILES = {"_full_train.log", "_app.log", "demo.png"}
EXCLUDE_PREFIXES = ("_",)


def keep(path: Path) -> bool:
    parts = path.parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    if path.name in EXCLUDE_FILES:
        return False
    if path.name.startswith(EXCLUDE_PREFIXES) and path.suffix in (".png", ".log", ".py"):
        return False
    if path.suffix in (".pyc",):
        return False
    return True


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file() or not keep(p):
                continue
            z.write(p, f"tholscript/{p.relative_to(ROOT)}")
            n += 1
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    main()
