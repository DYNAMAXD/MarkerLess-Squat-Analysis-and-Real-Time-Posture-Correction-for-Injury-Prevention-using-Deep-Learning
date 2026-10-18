"""
run_ablation.py - runs one model under several ablation variants (each = config overrides), one sub-folder per
variant, in its own process (started by notebook 03 through squat_research.monitor.launch_ablation).

    C:\\SquatModelVenv\\Scripts\\python.exe scripts\\run_ablation.py --model bigru_attn --variants variants.json --out results\\03_ablation\\runs

variants.json: {"full": {}, "no_velocity": {"data.velocity": false}, ...}
Writes <out>/ablation_state.json (variants, done, current) for monitoring; a variant whose summary.json exists is skipped,
so the run can be restarted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from squat_research.common import apply_overrides, load_config, write_json   # noqa: E402
from squat_research.dataset import build_dataset, dataset_file, load_dataset   # noqa: E402
from squat_research.engine import run_experiment   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    variants = json.loads(Path(a.variants).read_text(encoding="utf-8"))
    base = load_config(a.config)
    state = dict(model=a.model, variants=list(variants), done=[], current=None, state="running",
                 started=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(out / "ablation_state.json", state)
    try:
        for name, over in variants.items():
            vdir = out / name
            if (vdir / a.model / "summary.json").exists():
                state["done"].append(name)
                continue
            state["current"] = name
            write_json(out / "ablation_state.json", state)
            cfg = apply_overrides(base, over)
            if not dataset_file(cfg).exists():            # T32 / T128 / no_trim need their own dataset file
                build_dataset(cfg)
            run_experiment([a.model], load_dataset(cfg), cfg, vdir, save_models=True)
            write_json(vdir / "overrides.json", over)
            state["done"].append(name)
            write_json(out / "ablation_state.json", state)
        state.update(state="done", current=None)
    except BaseException:
        state["state"] = "error"
        raise
    finally:
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(out / "ablation_state.json", state)


if __name__ == "__main__":
    main()
