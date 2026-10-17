"""
run_ablation.py   (SCRIPT)
==========================
Step 4: ablation study. Re-trains ONE model (default: the transformer) with one component
removed or changed at a time, and compares each variant with the full model.

Run it yourself (from the squat_error_study folder):
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_ablation.py                       # all variants
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_ablation.py --epochs 3 --splits 3  # quick check
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_ablation.py --model bigru_attn     # ablate another model
    C:\\squat_venv\\Scripts\\python.exe scripts\\run_ablation.py --variants full no_velocity mean_pooling

The T32 / T128 variants need  scripts\\build_dataset.py --T 32  and  --T 128  to be run first
(otherwise they are skipped with a message).
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from squat_study import data as D          # noqa: E402
from squat_study import report as R        # noqa: E402
from squat_study import train as TR        # noqa: E402

# ----------------------------------------------------------------------------
# The variants. Keys are "section.setting"; every variant changes only what it lists.
# ----------------------------------------------------------------------------
VARIANTS = {
    "full":              {},
    # --- which inputs does the model need? ---
    "joints_only":       {"data.inputs": "joints"},
    "angles_only":       {"data.inputs": "frame"},
    "no_velocity":       {"data.velocity": False},
    "no_tempo":          {"data.tempo": False},
    # --- which camera? (simulated by dropping a body axis; angle features removed) ---
    "view_side_only":    {"data.view": "side", "data.inputs": "joints"},
    "view_front_only":   {"data.view": "front", "data.inputs": "joints"},
    # --- which model parts matter? (transformer only) ---
    "no_spatial_attn":   {"model.spatial_attn": False},
    "no_temporal_attn":  {"model.temporal_attn": False},
    "mean_pooling":      {"model.pooling": "mean"},
    "blocks_2":          {"model.n_blocks": 2},
    # --- training / data details ---
    "no_noise_aug":      {"train.noise_std": 0.0},
    "T32":               {"data.T": 32},
    "T128":              {"data.T": 128},
}
TRANSFORMER_ONLY = {"no_spatial_attn", "no_temporal_attn", "blocks_2", "mean_pooling"}   # settings other models ignore


def apply(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        sec, key = k.split(".")
        cfg[sec][key] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--model", default="transformer", choices=TR.NEURAL_MODELS)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--splits", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args()

    cfg0 = yaml.safe_load(open(args.config))
    if args.epochs: cfg0["train"]["epochs"] = args.epochs
    if args.splits: cfg0["cv"]["n_splits"] = args.splits
    if args.seeds: cfg0["cv"]["seeds"] = args.seeds

    device = TR.get_device()
    run_dir = ROOT / cfg0["paths"]["runs"] / f"{time.strftime('%Y%m%d-%H%M%S')}_ablation_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg0, open(run_dir / "config_used.yaml", "w"))
    print(f"device: {device}   model: {args.model}\nrun folder: {run_dir}\n")

    for name in (args.variants or list(VARIANTS)):
        if name in TRANSFORMER_ONLY and args.model != "transformer":
            print(f"--- {name}: skipped (transformer-only)"); continue
        cfg = apply(cfg0, VARIANTS[name])
        npz = ROOT / cfg["paths"]["processed"] / f"reps_T{cfg['data']['T']}.npz"
        if not npz.exists():
            print(f"--- {name}: skipped ({npz.name} missing - run build_dataset.py --T {cfg['data']['T']})")
            continue
        ds = D.load_dataset(npz, cfg["data"]["classes"], cfg["data"])
        print(f"=== {name}  {VARIANTS[name]}")
        TR.run_experiment(args.model, name, ds, cfg, run_dir, device, save_models=False)
        R.make_report(run_dir, ablation=True)

    print("\nFinished. Figures:", run_dir / "figures", " Tables:", run_dir / "tables")


if __name__ == "__main__":
    main()
