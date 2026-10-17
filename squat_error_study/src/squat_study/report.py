"""
report.py   (MODULE)
====================
Turns a run folder (results/runs/<run>/<experiment>/summary.json, preds.csv, ...) into
tables and PNG figures that can be pasted into slides or the paper.

Everything is written to <run>/figures/ and <run>/tables/.

Sections
--------
 1. Loading
 2. Tables            model comparison, per-class F1 (CSV + markdown)
 3. Figures           comparison bars, confusion matrices, per-class heatmap, training curves,
                      frame-attention (alpha) curves, ablation deltas
 4. make_report       does all of the above for one run folder
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
PRETTY = {"decision_tree": "Decision tree", "logreg": "Logistic regression",
          "random_forest": "Random forest", "xgboost": "XGBoost", "cnn1d": "1D-CNN",
          "bigru_attn": "BiGRU+attn", "bilstm": "BiLSTM", "stgcn": "ST-GCN",
          "transformer": "Transformer (proposed)"}


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_summaries(run_dir):
    out = {}
    for f in sorted(Path(run_dir).glob("*/summary.json")):
        out[f.parent.name] = json.load(open(f))
    return out


def _dirs(run_dir):
    run_dir = Path(run_dir)
    fig, tab = run_dir / "figures", run_dir / "tables"
    fig.mkdir(exist_ok=True); tab.mkdir(exist_ok=True)
    return fig, tab


def _label(exp, s):
    return PRETTY.get(exp, exp)


# ----------------------------------------------------------------------------
# 2. Tables
# ----------------------------------------------------------------------------
def comparison_table(summ, run_dir):
    fig, tab = _dirs(run_dir)
    rows = []
    for exp, s in summ.items():
        rows.append({"model": _label(exp, s),
                     "macro-F1 (rep)": f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f}",
                     "accuracy (rep)": f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}",
                     "macro-F1 (recording)": f"{s['rec_macro_f1_mean']:.3f} ± {s['rec_macro_f1_std']:.3f}",
                     "params": s.get("n_params") or "-", "seconds": s["seconds"],
                     "_sort": s["macro_f1_mean"]})
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    df.to_csv(tab / "model_comparison.csv", index=False)
    try:
        (tab / "model_comparison.md").write_text(df.to_markdown(index=False))
    except ImportError:                                  # tabulate is optional
        (tab / "model_comparison.md").write_text(df.to_string(index=False))
    return df


def per_class_table(summ, run_dir):
    _, tab = _dirs(run_dir)
    classes = next(iter(summ.values()))["classes"]
    df = pd.DataFrame({_label(e, s): s["per_class_f1_pooled"] for e, s in summ.items()}, index=classes).T
    df.round(3).to_csv(tab / "per_class_f1.csv")
    return df


# ----------------------------------------------------------------------------
# 3. Figures
# ----------------------------------------------------------------------------
def plot_comparison(summ, run_dir):
    fig_dir, _ = _dirs(run_dir)
    items = sorted(summ.items(), key=lambda kv: kv[1]["macro_f1_mean"])
    names = [_label(e, s) for e, s in items]
    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(items) + 1.5))
    ax.barh(y + 0.2, [s["macro_f1_mean"] for _, s in items], 0.4,
            xerr=[s["macro_f1_std"] for _, s in items], label="per rep", color="#3b6fb6", capsize=2)
    ax.barh(y - 0.2, [s["rec_macro_f1_mean"] for _, s in items], 0.4,
            xerr=[s["rec_macro_f1_std"] for _, s in items], label="per recording (vote)",
            color="#e0903a", capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("macro-F1 (mean ± std over folds, person-wise split)")
    ax.set_xlim(0, 1); ax.legend(loc="lower right"); ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(fig_dir / "model_comparison.png"); plt.close(fig)


def plot_confusions(summ, run_dir):
    fig_dir, _ = _dirs(run_dir)
    n = len(summ)
    cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    for ax, (exp, s) in zip(axes.ravel(), summ.items()):
        cm = np.array(s["confusion_pooled"], dtype=float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        k = len(s["classes"])
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        short = [c.replace("_", "\n") for c in s["classes"]]
        ax.set_xticklabels(short, fontsize=6); ax.set_yticklabels(short, fontsize=6)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_title(_label(exp, s), fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrices.png"); plt.close(fig)


def plot_per_class(df, run_dir):
    fig_dir, _ = _dirs(run_dir)
    fig, ax = plt.subplots(figsize=(1.2 * df.shape[1] + 2.5, 0.5 * df.shape[0] + 1.5))
    im = ax.imshow(df.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels([c.replace("_", "\n") for c in df.columns], fontsize=8)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="F1"); ax.set_title("Per-class F1 (pooled over folds)")
    fig.tight_layout(); fig.savefig(fig_dir / "per_class_f1.png"); plt.close(fig)


def plot_training_curves(run_dir, exp):
    fig_dir, _ = _dirs(run_dir)
    hs = [json.load(open(f)) for f in sorted((Path(run_dir) / exp).glob("fold_*/history.json"))]
    if not hs:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for h in hs:
        a.plot(h["train_loss"], alpha=0.6); b.plot(h["val_f1"], alpha=0.6)
    a.set_title("training loss"); a.set_xlabel("epoch")
    b.set_title("validation macro-F1 (held-out persons)"); b.set_xlabel("epoch"); b.set_ylim(0, 1)
    fig.suptitle(PRETTY.get(exp, exp)); fig.tight_layout()
    fig.savefig(fig_dir / f"training_curves_{exp}.png"); plt.close(fig)


def plot_alpha(run_dir, exp, classes):
    """Mean frame-attention alpha per true class over normalised rep time (0 = start, 1 = end)."""
    fig_dir, _ = _dirs(run_dir)
    ap = Path(run_dir) / exp / "alpha.npy"
    if not ap.exists():
        return
    alpha = np.load(ap)
    preds = pd.read_csv(Path(run_dir) / exp / "preds.csv")
    if len(preds) != len(alpha):
        return
    x = np.linspace(0, 1, alpha.shape[1])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, c in enumerate(classes):
        a = alpha[preds["y"].to_numpy() == i]
        if len(a):
            m, s = a.mean(0), a.std(0)
            ax.plot(x, m, label=c); ax.fill_between(x, m - s, m + s, alpha=0.12)
    ax.axhline(1 / alpha.shape[1], color="grey", ls=":", lw=1, label="uniform")
    ax.set_xlabel("normalised rep time (start -> bottom -> end)"); ax.set_ylabel("mean alpha")
    ax.set_title("Which frames mattered (frame attention), per true class")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(fig_dir / f"alpha_{exp}.png"); plt.close(fig)


def plot_ablation(summ, run_dir, baseline="full"):
    fig_dir, tab = _dirs(run_dir)
    if baseline not in summ:
        return
    base = summ[baseline]["macro_f1_mean"]
    rows = [(e, s["macro_f1_mean"], s["macro_f1_std"]) for e, s in summ.items()]
    df = pd.DataFrame(rows, columns=["variant", "macro_f1", "std"])
    df["delta_vs_full"] = df["macro_f1"] - base
    df.sort_values("macro_f1").to_csv(tab / "ablation.csv", index=False)
    df = df.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(df) + 1.5))
    colors = ["#c0504d" if v != baseline else "#3b6fb6" for v in df["variant"]]
    ax.barh(df["variant"], df["macro_f1"], xerr=df["std"], color=colors, capsize=2)
    ax.axvline(base, color="#3b6fb6", ls="--", lw=1)
    for i, (v, d) in enumerate(zip(df["variant"], df["delta_vs_full"])):
        ax.text(0.01, i, f"{d:+.3f}", va="center", fontsize=8, color="white")
    ax.set_xlim(0, 1); ax.set_xlabel("macro-F1 (mean ± std over folds)")
    ax.set_title("Ablation study (numbers = change vs full model)")
    fig.tight_layout(); fig.savefig(fig_dir / "ablation.png"); plt.close(fig)


# ----------------------------------------------------------------------------
# 4. make_report
# ----------------------------------------------------------------------------
def make_report(run_dir, ablation=False):
    run_dir = Path(run_dir)
    summ = load_summaries(run_dir)
    if not summ:
        print("no summaries in", run_dir); return
    comparison_table(summ, run_dir)
    if ablation:
        plot_ablation(summ, run_dir)
    else:
        plot_comparison(summ, run_dir)
        plot_confusions(summ, run_dir)
        plot_per_class(per_class_table(summ, run_dir), run_dir)
    classes = next(iter(summ.values()))["classes"]
    for exp in summ:
        plot_training_curves(run_dir, exp)
        plot_alpha(run_dir, exp, classes)
    print("report written to", run_dir / "figures", "and", run_dir / "tables")
