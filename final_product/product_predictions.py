"""
product_predictions.py -- the predictions.json contract of the final product (see model_build_instructions.md, section 9).

The trained squat model writes predictions.json itself (squat_model.predict). Until it exists, the product writes
  - "inactive" predictions: the reps found by the preprocessing, without errors (model_active = false), or
  - demo predictions on request: made-up errors and attention so the whole page can be tried out (demo = true;
    the page shows a DEMO banner). They say nothing about the squat.

Also writes rep_results.csv, the user-facing reps.csv (rep, start, end in seconds) and the session summary.

    python final_product/product_predictions.py <session folder> [--demo]
"""
import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np

CLASSES = ["correct", "insufficient_depth", "forward_lean", "knee_valgus", "knee_varus", "heel_lift", "abnormal_tempo"]
UNCERTAIN_BELOW = 0.5
WINDOW = 10      # frames: attention is averaged in windows of this many frames ...
SIGMA = 5.0      # ... then Gaussian-smoothed (frames)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def session_reps(folder):
    """Reps of a preprocessed recording: frames from reps.csv, bottom from results.json (seconds -> frame)."""
    folder = Path(folder)
    res = load_json(folder / "results.json", {})
    fps = float((res.get("fps") or {}).get("p3d") or 29.97)
    bottoms = {int(r["id"]): r["t"][1] for r in res.get("reps") or []}
    reps = []
    p = folder / "reps.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                n, a, b = int(row["rep_no"]), int(row["start_frame"]), int(row["end_frame"])
                bot = int(round(bottoms[n] * fps)) if n in bottoms else (a + b) // 2
                reps.append({"rep": n, "start_frame": a, "end_frame": b, "bottom_frame": min(max(bot, a), b)})
    else:  # older outputs: seconds only
        for r in res.get("reps") or []:
            a, bot, b = (int(round(s * fps)) for s in r["t"])
            reps.append({"rep": int(r["id"]), "start_frame": a, "end_frame": b, "bottom_frame": bot})
    n_frames = int((res.get("p3d") or {}).get("n_frames") or (max([r["end_frame"] for r in reps], default=-1) + 1))
    return reps, fps, n_frames


def alpha_to_frames(alpha, n):
    """Model attention (T values) -> one value per real frame of the rep: interpolate, average in 10-frame windows,
    Gaussian-smooth (sigma 5 frames), peak = 1."""
    alpha = np.asarray(alpha, dtype=float)
    if n <= 0 or alpha.size == 0:
        return []
    x = np.interp(np.linspace(0, alpha.size - 1, n), np.arange(alpha.size), alpha)
    for i in range(0, n, WINDOW):
        x[i:i + WINDOW] = x[i:i + WINDOW].mean()
    r = int(3 * SIGMA)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / SIGMA) ** 2)
    x = np.convolve(np.pad(x, r, mode="edge"), k / k.sum(), mode="valid")
    m = x.max()
    return [round(float(v), 4) for v in (x / m if m > 0 else x)]


def assemble(folder, reps, *, model_active, demo=False, model_version=None):
    """reps: dicts with rep, start_frame, end_frame (+ label, confidence, probs, alpha, valid) -> predictions dict."""
    _, fps, n_frames = session_reps(folder)
    intensity = np.zeros(n_frames)
    out = []
    for r in reps:
        r = dict(r)
        r["start_s"], r["end_s"] = round(r["start_frame"] / fps, 3), round(r["end_frame"] / fps, 3)
        r.setdefault("valid", True)
        r.setdefault("label", None)
        if r["label"] is not None:
            r["uncertain"] = r["confidence"] < UNCERTAIN_BELOW
            n = r["end_frame"] - r["start_frame"] + 1
            r["alpha_frames"] = alpha_to_frames(r["alpha"], n)
            if r["label"] != "correct":
                a, b = r["start_frame"], min(r["end_frame"], n_frames - 1)
                if b >= a:
                    intensity[a:b + 1] = np.asarray(r["alpha_frames"][:b - a + 1]) * r["confidence"]
        else:
            r["uncertain"] = False
        out.append(r)
    return {"model_version": model_version, "model_active": model_active, "demo": demo, "classes": CLASSES,
            "fps": fps, "n_frames": n_frames, "uncertain_below": UNCERTAIN_BELOW,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "reps": out,
            "intensity": [round(float(v), 4) for v in intensity]}


def inactive_predictions(folder):
    reps, _, _ = session_reps(folder)
    return assemble(folder, [{k: r[k] for k in ("rep", "start_frame", "end_frame")} for r in reps], model_active=False)


def demo_predictions(folder, seed=7):
    """Made-up errors + attention peaking around the bottom of each rep. For trying the page only."""
    reps, _, _ = session_reps(folder)
    rng = np.random.default_rng(seed)
    errors = CLASSES[1:]
    main_err = errors[int(rng.integers(len(errors)))]
    out = []
    for i, r in enumerate(reps):
        roll = rng.random()
        label = "correct" if roll < 0.45 else main_err if roll < 0.85 else errors[int(rng.integers(len(errors)))]
        conf = float(rng.uniform(0.38, 0.95) if i % 5 == 3 else rng.uniform(0.62, 0.95))
        probs = rng.dirichlet(np.ones(len(CLASSES))) * (1 - conf)
        probs[CLASSES.index(label)] = conf
        probs = probs / probs.sum()
        T = 64
        n = max(1, r["end_frame"] - r["start_frame"])
        peak = (r["bottom_frame"] - r["start_frame"]) / n * (T - 1) + rng.normal(0, 3)
        alpha = np.exp(-0.5 * ((np.arange(T) - peak) / rng.uniform(5, 9)) ** 2) + 0.03
        out.append({"rep": r["rep"], "start_frame": r["start_frame"], "end_frame": r["end_frame"], "label": label,
                    "confidence": round(float(probs[CLASSES.index(label)]), 3),
                    "probs": [round(float(p), 4) for p in probs], "alpha": [round(float(a), 5) for a in alpha / alpha.sum()]})
    return assemble(folder, out, model_active=False, demo=True, model_version="demo")


def rep_results_csv(pred):
    cols = ["rep", "start_s", "end_s", "start_frame", "end_frame", "duration_s", "predicted_error", "confidence",
            *[f"p_{c}" for c in CLASSES], "alpha_peak_s", "uncertain"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in pred["reps"]:
        probs = r.get("probs") or [None] * len(CLASSES)
        peak = None
        if r.get("alpha_frames"):
            peak = round(r["start_s"] + int(np.argmax(r["alpha_frames"])) / pred["fps"], 3)
        w.writerow([r["rep"], r["start_s"], r["end_s"], r["start_frame"], r["end_frame"], round(r["end_s"] - r["start_s"], 3),
                    r.get("label") or "", r.get("confidence", ""), *["" if p is None else p for p in probs],
                    "" if peak is None else peak, r.get("uncertain", "")])
    return buf.getvalue()


def user_reps_csv(pred):
    """The user-facing per-rep file: rep, start, end (seconds on the shared timeline)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["rep", "start", "end"])
    for r in pred["reps"]:
        w.writerow([r["rep"], r["start_s"], r["end_s"]])
    return buf.getvalue()


def summary(folder, pred):
    """Numbers for the session header and the history page."""
    res = load_json(Path(folder) / "results.json", {})
    analysed = [r for r in pred["reps"] if r.get("label")]
    counts = {c: sum(r["label"] == c for r in analysed) for c in CLASSES}
    errs = {c: n for c, n in counts.items() if c != "correct" and n}
    depth = [t.get("squat_depth_knee_deg_bottom") for t in res.get("table") or []]
    depth = [d for d in depth if isinstance(d, (int, float))]
    durs = [r["end_s"] - r["start_s"] for r in pred["reps"]]
    return {"reps": len(pred["reps"]), "analysed": len(analysed), "counts": counts,
            "pct_correct": round(100 * counts["correct"] / len(analysed)) if analysed else None,
            "top_error": max(errs, key=errs.get) if errs else None,
            "mean_depth_deg": round(float(np.mean(depth)), 1) if depth else None,
            "mean_rep_s": round(float(np.mean(durs)), 2) if durs else None,
            "model_active": pred.get("model_active", False), "demo": pred.get("demo", False)}


def write_all(folder, pred):
    folder = Path(folder)
    (folder / "predictions.json").write_text(json.dumps(pred), encoding="utf-8")
    (folder / "rep_results.csv").write_text(rep_results_csv(pred), encoding="utf-8")
    return summary(folder, pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--demo", action="store_true", help="made-up predictions for trying the page")
    a = ap.parse_args()
    pred = demo_predictions(a.folder) if a.demo else inactive_predictions(a.folder)
    print(json.dumps(write_all(a.folder, pred), indent=1))


if __name__ == "__main__":
    main()
