import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tholscript.scripts.train_crnn import main

def run():
    t0 = time.time()
    for script in ("tamil", "brahmi"):
        main(script)
    print(f"CRNN TOTAL {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    run()
