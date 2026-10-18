"""
run_notebooks.py - executes notebooks in order (outputs saved inside each notebook), stops at the first failure.

    C:\\SquatModelVenv\\Scripts\\python.exe scripts\\run_notebooks.py 01 02 03 04
Progress is appended to results/run_notebooks_log.txt.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "run_notebooks_log.txt"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main(prefixes):
    for pre in prefixes:
        nb = next((ROOT / "notebooks").glob(f"{pre}_*.ipynb"))
        log(f"start {nb.name}")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=squatmodel", nb.name],
                           cwd=str(ROOT / "notebooks"), capture_output=True, text=True)
        if r.returncode != 0:
            log(f"FAILED {nb.name} after {time.time() - t0:.0f} s:\n{r.stderr[-3000:]}")
            sys.exit(1)
        log(f"done {nb.name} in {(time.time() - t0) / 60:.1f} min")
    log("all done")


if __name__ == "__main__":
    main(sys.argv[1:] or ["01", "02", "03", "04"])
