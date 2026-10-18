"""
report.py - one display-ready report per model: what it receives, how it processes it, how it performs.

    md = model_report(run_dir, "tcn", ds, cfg, out_folder)    # writes out_folder/<model>/report.md + figures, returns markdown
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import viz
from .common import nice, read_json
from .models import MODEL_INFO, TREE_MODELS, build_model, n_params


def model_report(run_dir, model, ds, cfg, out_folder):
    run_dir, out = Path(run_dir), Path(out_folder) / model
    out.mkdir(parents=True, exist_ok=True)
    s = read_json(run_dir / model / "summary.json")
    info = MODEL_INFO[model]
    C = ds["classes"]
    T, J, F = ds["joints"].shape[1], ds["joints"].shape[2], ds["frame"].shape[2]
    lines = [f"## {nice(model)}  ({info['family']})", "",
             "**Input.** " + info["input"] + ".", "",
             "**Processing.** " + info["processing"] + ".", ""]
    if model not in TREE_MODELS:
        m = build_model(model, J, F, T, len(C), ds["meta"]["joint_names"], cfg["model"])
        lines += [f"**Size.** {n_params(m):,} trainable parameters. Training: AdamW (lr {cfg['train']['lr']}, weight decay "
                  f"{cfg['train']['weight_decay']}), cosine schedule, up to {cfg['train']['epochs']} epochs, early stopping on the "
                  f"validation persons (patience {cfg['train']['patience']}), label smoothing {cfg['train']['label_smoothing']}, "
                  f"noise augmentation {cfg['train']['noise_std']}.", ""]
    if model in TREE_MODELS:
        lines += ["**Preprocessing.** The summary row is computed by the Pose2Sim preprocessing pipeline per rep (`rep_features.csv`, "
                  "statistics of its joint angles); the 4 tempo numbers come from the trimmed rep. No body-frame joints are used.", ""]
    else:
        lines += [f"**Common preprocessing.** 3D joints from Pose2Sim -> body frame (lateral / up / forward from the standing posture and the feet), "
                  f"hip-centred, divided by the leg length; reps trimmed to the movement and resampled to {T} frames; z-scored with "
                  "training-fold statistics.", ""]
    pc = pd.DataFrame({"class": [c.replace("_", " ") for c in C], "F1": s["per_class_f1"],
                       "reps": np.array(s["confusion"]).sum(1)})
    lines += ["**Performance (5-fold cross-validation by person x 3 seeds = 15 runs).**", "",
              "| metric | value |", "|---|---|",
              f"| macro-F1 per rep | {s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} |",
              f"| accuracy per rep | {s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} |",
              f"| macro-F1 per recording (vote) | {s['rec_macro_f1_mean']:.3f} ± {s['rec_macro_f1_std']:.3f} |",
              f"| best / worst run | {max(s['fold_f1']):.3f} / {min(s['fold_f1']):.3f} |"]
    if s.get("alpha_drop") is not None:
        lines.append(f"| alpha check: drop when the top-10 % alpha frames are masked vs random | {s['alpha_drop']:.3f} vs {s['random_drop']:.3f} |")
    if s["best_epochs"]:
        lines.append(f"| median best epoch | {int(np.median(s['best_epochs']))} |")
    lines.append(f"| training time (15 runs) | {s['seconds'] / 60:.1f} min |")
    lines += ["", "**F1 per class.**", "", pc.to_markdown(index=False, floatfmt=".3f"), ""]
    weakest = pc.sort_values("F1").iloc[0]
    cm = np.array(s["confusion"], float)
    np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    lines += [f"Weakest class: **{weakest['class']}** (F1 {weakest['F1']:.2f}). Most frequent confusion: "
              f"**{C[i].replace('_', ' ')} -> {C[j].replace('_', ' ')}** ({int(cm[i, j])} reps).", ""]
    figs = [viz.confusion_single(run_dir, model, C, out, "confusion")]
    if model not in TREE_MODELS:
        figs.append(viz.learning_curves(run_dir, model, out))
        figs.append(viz.alpha_profile(run_dir, model, C, out))
    lines += ["![confusion](confusion.png)"]
    if model not in TREE_MODELS:
        lines += [f"![learning curves](learning_curves_{model}.png)", f"![alpha profile](alpha_profile_{model}.png)"]
    md = "\n".join(lines)
    (out / "report.md").write_text(md, encoding="utf-8")
    import matplotlib.pyplot as plt
    for f in figs:
        if f is not None:
            plt.close(f)
    return md
