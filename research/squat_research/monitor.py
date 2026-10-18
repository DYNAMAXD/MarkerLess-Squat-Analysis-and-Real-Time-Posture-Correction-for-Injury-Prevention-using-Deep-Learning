"""
monitor.py - start an experiment in the background from a notebook and watch it.

    proc = launch(["bigru_attn", "st_transformer_spec"], "results/02_attention_models/cv")
    watch("results/02_attention_models/cv")          # live table, refreshes until the run is done (interrupt = stop watching)
    status("results/02_attention_models/cv")         # one snapshot
    stop("results/02_attention_models/cv")           # stop the run

The run keeps going if the notebook kernel is restarted: watch() just reads progress.json and log.txt.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .common import ROOT, nice, read_json

PY = sys.executable


def _dir(out):
    out = Path(out)
    return out if out.is_absolute() else ROOT / out


def launch(models, out, config=None, overrides=None, save_models=False):
    """Start scripts/run_models.py in its own process (no window). Returns the Popen; its pid is saved in run.pid."""
    out = _dir(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", str(ROOT / "scripts" / "run_models.py"), "--out", str(out), "--models", *models]
    if config:
        cmd += ["--config", str(config)]
    if overrides:
        cmd += ["--set", *[f"{k}={json.dumps(v)}" for k, v in overrides.items()]]
    if save_models:
        cmd.append("--save-models")
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    console = open(out / "console.txt", "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=console, stderr=subprocess.STDOUT, creationflags=flags)
    (out / "run.pid").write_text(str(p.pid))
    print(f"started pid {p.pid}: {len(models)} model(s) -> {out}")
    return p


def _alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _fmt(s):
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def status(out, log_lines=8):
    """Text snapshot of a run: state, folds done, ETA, current epoch, per-model mean F1 so far, last log lines."""
    out = _dir(out)
    st = read_json(out / "progress.json")
    if st is None:
        return f"no progress.json in {out} yet"
    pid = (out / "run.pid").read_text().strip() if (out / "run.pid").exists() else None
    state = st["state"]
    if state == "running" and pid and not _alive(pid):
        state = "stopped / crashed (process gone) - see console.txt"
    done = st["done"]
    lines = [f"state: {state}   folds {len(done)}/{st['total_folds']}   elapsed {_fmt(st.get('elapsed_s'))}   "
             f"ETA {_fmt(st.get('eta_s'))}   updated {st['updated']}"]
    c = st.get("current")
    if c and state == "running":
        vf = "-" if c["val_f1"] is None else f"{c['val_f1']:.3f}"
        lines.append(f"now: {nice(c['model'])}  seed {c['seed']}  fold {c['fold']}  epoch {c['epoch']}/{c['max_epochs']}  "
                     f"train loss {c['train_loss']:.3f}  val F1 {vf}  best epoch {c['best_epoch']}")
    if done:
        df = pd.DataFrame(done).groupby("model")["f1"].agg(["count", "mean", "std"]).reset_index()
        df["model"] = df["model"].map(nice)
        lines.append("\n" + df.rename(columns={"count": "folds done", "mean": "macro-F1 so far", "std": "std"})
                     .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log = out / "log.txt"
    if log.exists():
        lines.append("\n" + "\n".join(log.read_text(encoding="utf-8").splitlines()[-log_lines:]))
    return "\n".join(lines)


def watch(out, every=20, log_lines=8):
    """Refreshing view in a notebook until the run ends. Interrupting the cell stops watching, not the run."""
    from IPython.display import clear_output
    out = _dir(out)
    try:
        while True:
            txt = status(out, log_lines)
            clear_output(wait=True)
            print(txt)
            st = read_json(out / "progress.json") or {}
            pid = (out / "run.pid").read_text().strip() if (out / "run.pid").exists() else None
            if st.get("state") in ("done", "error") or (pid and not _alive(pid) and st):
                break
            time.sleep(every)
    except KeyboardInterrupt:
        print("stopped watching (the run continues)")


def wait(out, every=30):
    """Block until the run ends (for unattended notebook execution); prints a short line now and then."""
    out = _dir(out)
    last = 0
    while True:
        st = read_json(out / "progress.json") or {}
        pid = (out / "run.pid").read_text().strip() if (out / "run.pid").exists() else None
        if st.get("state") in ("done", "error") or (pid and not _alive(pid) and st):
            print(status(out, 3))
            return st.get("state")
        if time.time() - last > 600:
            print(time.strftime("%H:%M"), f"folds {len(st.get('done', []))}/{st.get('total_folds', '?')}, ETA {_fmt(st.get('eta_s'))}", flush=True)
            last = time.time()
        time.sleep(every)


def launch_ablation(model, variants, out, config=None):
    """Start scripts/run_ablation.py in its own process; variants = {name: {"section.key": value}}."""
    out = _dir(out)
    out.mkdir(parents=True, exist_ok=True)
    vf = out / "variants.json"
    vf.write_text(json.dumps(variants, indent=1), encoding="utf-8")
    cmd = [PY, "-u", str(ROOT / "scripts" / "run_ablation.py"), "--model", model, "--variants", str(vf), "--out", str(out)]
    if config:
        cmd += ["--config", str(config)]
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    console = open(out / "console.txt", "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=console, stderr=subprocess.STDOUT, creationflags=flags)
    (out / "run.pid").write_text(str(p.pid))
    print(f"started pid {p.pid}: {model}, {len(variants)} variants -> {out}")
    return p


def ablation_status(out, log_lines=4):
    """Variants done / total + the status of the variant running now."""
    out = _dir(out)
    st = read_json(out / "ablation_state.json")
    if st is None:
        return "no ablation_state.json yet"
    pid = (out / "run.pid").read_text().strip() if (out / "run.pid").exists() else None
    state = st["state"] if not (st["state"] == "running" and pid and not _alive(pid)) else "stopped / crashed - see console.txt"
    txt = [f"ablation of {nice(st['model'])}: {state}, variants {len(st['done'])}/{len(st['variants'])}; now: {st.get('current')}"]
    rows = []
    for v in st["done"]:
        s = read_json(out / v / st["model"] / "summary.json")
        if s:
            rows.append(f"  {v:<22} macro-F1 {s['macro_f1_mean']:.3f} +- {s['macro_f1_std']:.3f}")
    txt += rows
    if st.get("current"):
        txt.append("\n" + status(out / st["current"], log_lines))
    return "\n".join(txt)


def wait_ablation(out, every=30):
    out = _dir(out)
    last = 0
    while True:
        st = read_json(out / "ablation_state.json") or {}
        pid = (out / "run.pid").read_text().strip() if (out / "run.pid").exists() else None
        if st.get("state") in ("done", "error") or (pid and not _alive(pid) and st):
            print(ablation_status(out, 2))
            return st.get("state")
        if time.time() - last > 600:
            print(time.strftime("%H:%M"), f"variants {len(st.get('done', []))}/{len(st.get('variants', []))}, now {st.get('current')}", flush=True)
            last = time.time()
        time.sleep(every)


def stop(out):
    out = _dir(out)
    pid = (out / "run.pid").read_text().strip() if (out / "run.pid").exists() else None
    if pid and _alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", pid], capture_output=True)
        else:
            os.kill(int(pid), 9)
        print("stopped", pid)
    else:
        print("not running")
