"""
run_models.py - runs person-wise cross-validation of a list of models in its own process (started by the notebooks
with squat_research.monitor.launch, or by hand), writing progress.json / log.txt for monitoring.

    C:\\SquatModelVenv\\Scripts\\python.exe scripts\\run_models.py --out results\\02_attention_models\\cv ^
        --models bigru_attn st_transformer_lite [--config config\\default.yaml] [--set train.epochs=30 ...] [--save-models]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from squat_research.common import load_config   # noqa: E402
from squat_research.dataset import load_dataset   # noqa: E402
from squat_research.engine import run_experiment   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help='overrides "section.key=json value"')
    ap.add_argument("--save-models", action="store_true")
    a = ap.parse_args()
    over = {}
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            over[k] = json.loads(v)
        except ValueError:
            over[k] = v
    cfg = load_config(a.config, over)
    ds = load_dataset(cfg)
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    run_experiment(a.models, ds, cfg, out, save_models=a.save_models)


if __name__ == "__main__":
    main()
