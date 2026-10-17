"""
run_cv.py   (SCRIPT)
====================
Step 3: person-wise cross-validation of every model + the comparison figures.

Run it yourself (from the squat_error_study folder):
    # everything in configs/default.yaml  (long: use the lab GPU)
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_cv.py

    # quick check that everything works (2-3 minutes on CPU)
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_cv.py --models decision_tree transformer --epochs 3 --splits 3

    # only the cheap baselines (about a minute)
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_cv.py --models decision_tree logreg random_forest xgboost

    # three seeds for the paper numbers
    C:\squat_venv\Scripts\python.exe scripts\run_cv.py --seeds 0 1 2

    # also fit one final model on ALL persons and save it (for the product)
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_cv.py --models transformer --final transformer

Everything is saved in results/runs/<timestamp>_cv/ (tables/, figures/, one folder per model).
"""

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from squat_study import data as D          # noqa: E402
from squat_study import report as R        # noqa: E402
from squat_study import train as TR        # noqa: E402


def load_cfg(path, args):
    cfg = yaml.safe_load(open(path))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.splits:
        cfg["cv"]["n_splits"] = args.splits
    if args.seeds:
        cfg["cv"]["seeds"] = args.seeds
    if args.T:
        cfg["data"]["T"] = args.T
    if args.models:
        cfg["run_models"] = args.models
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--models", nargs="+", choices=TR.ALL_MODELS)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--splits", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--T", type=int)
    ap.add_argument("--final", choices=TR.ALL_MODELS, help="after CV, train this model on all persons and save it")
    ap.add_argument("--name", default="cv")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args)
    device = TR.get_device()
    npz = ROOT / cfg["paths"]["processed"] / f"reps_T{cfg['data']['T']}.npz"
    if not npz.exists():
        sys.exit(f"{npz} not found - run scripts/build_dataset.py first")
    ds = D.load_dataset(npz, cfg["data"]["classes"], cfg["data"])

    run_dir = ROOT / cfg["paths"]["runs"] / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config_used.yaml", "w"))
    print(f"device: {device}   reps: {len(ds['y'])}   persons: {len(set(ds['person']))}   "
          f"classes: {ds['classes']}\nrun folder: {run_dir}\n")

    for name in cfg["run_models"]:
        print(f"=== {name}")
        TR.run_experiment(name, name, ds, cfg, run_dir, device)
        R.make_report(run_dir)                       # refresh tables/figures after every model

    if args.final:
        TR.train_final(args.final, ds, cfg, run_dir / f"final_{args.final}", device)

    print("\nFinished. Open", run_dir / "figures", "for the plots and", run_dir / "tables", "for the tables.")


if __name__ == "__main__":
    main()
