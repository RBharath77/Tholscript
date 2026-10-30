"""Background full-training entry point (used while packaging the project)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tholscript.scripts.train_all import run
t0 = time.time()
run(quick=False)
print(f"TOTAL {time.time()-t0:.0f}s", flush=True)
