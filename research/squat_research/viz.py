"""
viz.py - every figure of the notebooks. Each function draws, saves a PNG into `folder` (if given) and returns the figure.
Colours: base models orange, attention models blue; classes in a fixed categorical order (validated palette).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .common import nice, read_json
from .features import FRAME_FEATURE_NAMES

BONES = [("Hip", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RBigToe"), ("RAnkle", "RSmallToe"),
         ("RAnkle", "RHeel"), ("Hip", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LBigToe"),
         ("LAnkle", "LSmallToe"), ("LAnkle", "LHeel"), ("Hip", "Neck"), ("Neck", "Head"), ("Head", "Nose"),
         ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"), ("Neck", "LShoulder"),
         ("LShoulder", "LElbow"), ("LElbow", "LWrist")]          # for drawing skeletons
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed order, validated
BASE, ATTN = "#eb6834", "#2a78d6"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold"})


def save(fig, folder, name):
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(folder) / f"{name}.png", bbox_inches="tight")
    return fig


def pretty(c):
    return c.replace("_", " ")


# ================================================================================================= dataset
def class_counts(ds, folder=None):
    C = ds["classes"]
    reps = np.bincount(ds["y"], minlength=len(C))
    recs = [len(set(ds["recording"][ds["y"] == i])) for i in range(len(C))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].bar(range(len(C)), reps, color=CLASS_COLORS)
    ax[0].set_title("Reps per class")
    ax[1].bar(range(len(C)), recs, color=CLASS_COLORS)
    ax[1].set_title("Recordings per class")
    for a, v in zip(ax, (reps, recs)):
        a.set_xticks(range(len(C)), [pretty(c) for c in C], rotation=20, ha="right")
        for i, x in enumerate(v):
            a.text(i, x, str(x), ha="center", va="bottom", fontsize=9)
    return save(fig, folder, "dataset_class_counts")


def person_class_matrix(ds, folder=None):
    C = ds["classes"]
    persons = sorted(set(ds["person"]))
    M = np.array([[int(((ds["person"] == p) & (ds["y"] == i)).sum()) for i in range(len(C))] for p in persons])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(persons) + 1.2))
    sns.heatmap(M, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[pretty(c) for c in C], yticklabels=persons, ax=ax)
    ax.set_title("Reps per person and class (0 = recording missing or excluded)")
    ax.grid(False)
    return save(fig, folder, "dataset_person_class")


def flex_curves(ds, folder=None):
    """Mean knee flexion over the (trimmed, resampled) rep per class, +- 1 std."""
    C, T = ds["classes"], ds["flex"].shape[1]
    x = np.linspace(0, 100, T)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, c in enumerate(C):
        f = ds["flex"][ds["y"] == i]
        m, s = f.mean(0), f.std(0)
        ax.plot(x, m, color=CLASS_COLORS[i], lw=2, label=pretty(c))
        ax.fill_between(x, m - s, m + s, color=CLASS_COLORS[i], alpha=0.08)
    ax.set(xlabel="% of the rep (trimmed start -> bottom -> end)", ylabel="knee flexion (deg)", title="Knee flexion through the rep, per class")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    return save(fig, folder, "dataset_knee_flexion_curves")


def rep_features_table(ds):
    """Per-rep numbers that describe each error, from the frame features and tempo."""
    F = {n: i for i, n in enumerate(FRAME_FEATURE_NAMES)}
    fr = ds["frame"]
    return pd.DataFrame({
        "class": [pretty(c) for c in np.array(ds["classes"])[ds["y"]]],
        "max knee flexion (deg)": fr[:, :, [F["knee_flexion_L"], F["knee_flexion_R"]]].mean(2).max(1),
        "max trunk flexion (deg)": fr[:, :, F["sagittal_flexion_trunk"]].max(1),
        "max heel height (leg lengths)": fr[:, :, [F["heel_height_L"], F["heel_height_R"]]].max(2).max(1),
        "max knee offset, + = inward (leg lengths)": fr[:, :, [F["knee_offset_L"], F["knee_offset_R"]]].mean(2).max(1),
        "min knee offset, - = outward (leg lengths)": fr[:, :, [F["knee_offset_L"], F["knee_offset_R"]]].mean(2).min(1),
        "hip drop (leg lengths)": -fr[:, :, F["hip_height_rel"]].min(1),
        "rep duration (s)": ds["tempo"][:, 0],
    })


def feature_boxplots(ds, folder=None):
    df = rep_features_table(ds)
    cols = [c for c in df.columns if c != "class"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    order = [pretty(c) for c in ds["classes"]]
    for ax, c in zip(axes.flat, cols):
        sns.boxplot(data=df, x="class", y=c, hue="class", legend=False, order=order, hue_order=order, palette=CLASS_COLORS, ax=ax, fliersize=1.5, linewidth=0.8)
        ax.set(xlabel="", title=c)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("What separates the classes: one number per rep", fontweight="bold")
    fig.tight_layout()
    return save(fig, folder, "dataset_feature_boxplots")


def quality(ds, folder=None):
    rec = ds["meta"]["recordings"]
    tilt = [v["up_tilt_deg"] for v in rec.values()]
    leg = [v["leg_length_m"] for v in rec.values()]
    cnt = {}
    for v in rec.values():
        for k in v["missing_per_joint"]:
            cnt[k] = cnt.get(k, 0) + 1
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].hist(tilt, bins=20, color=ATTN)
    ax[0].set(title="World axis vs real standing 'up' (deg)", xlabel="tilt that the snapped axis would have had")
    ax[1].hist(leg, bins=20, color=ATTN)
    ax[1].set(title="Leg length per recording (m)")
    if cnt:
        s = pd.Series(cnt).sort_values()
        ax[2].barh(s.index, s.values, color=BASE)
    ax[2].set(title="Recordings with a joint missing > 5 % of frames")
    fig.tight_layout()
    return save(fig, folder, "dataset_quality")


def skeleton_example(ds, folder=None, cls=None):
    """Side and front view of one rep per class at its start and at its bottom (body frame, leg lengths)."""
    names = ds["meta"]["joint_names"]
    ix = {n: i for i, n in enumerate(names)}
    C = ds["classes"]
    fig, axes = plt.subplots(2, len(C), figsize=(2.6 * len(C), 5.6), sharex="row", sharey=True)
    for i, c in enumerate(C):
        k = np.where(ds["y"] == i)[0][0]
        J = ds["joints"][k, :, :, :3]
        bot = int(np.argmax(ds["flex"][k]))
        for row, (ax_h, title) in enumerate(((2, "side (forward / up)"), (0, "front (lateral / up)"))):
            ax = axes[row, i]
            for t, alpha in ((0, 0.3), (bot, 1.0)):
                P = J[t]
                for a, b in BONES:
                    if a in ix and b in ix and np.abs(P[ix[a]]).sum() > 0 and np.abs(P[ix[b]]).sum() > 0:
                        ax.plot([P[ix[a], ax_h], P[ix[b], ax_h]], [P[ix[a], 1], P[ix[b], 1]], color=CLASS_COLORS[i], alpha=alpha, lw=1.6)
            ax.set_aspect("equal")
            ax.set_title(f"{pretty(c)}\n{title}" if row == 0 else title, fontsize=8)
            ax.grid(False)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.tick_params(labelsize=7)
    fig.suptitle("Example rep per class: standing (faint) and bottom (solid), body frame", fontweight="bold")
    fig.tight_layout()
    return save(fig, folder, "dataset_skeleton_examples")


# ================================================================================================= results
def family(m):
    return "base" if m in ("logreg", "random_forest", "xgboost") else "attention"


def comparison(table, folder=None, name="comparison_macro_f1", title="Macro-F1 per rep (5-fold by person x 3 seeds)"):
    t = table.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2))
    cols = [BASE if family(m) == "base" else ATTN for m in t["model"]]
    ax.barh(t["name"], t["macro_f1"], xerr=t["macro_f1_std"], color=cols, height=0.6, error_kw=dict(ecolor="#555", capsize=3, lw=1))
    ax.scatter(t["rec_macro_f1"], t["name"], marker="D", s=22, color="#222", zorder=3, label="recording-level macro-F1")
    for y, v in enumerate(t["macro_f1"]):
        ax.text(0.01, y, f"{v:.3f}", va="center", ha="left", color="white", fontsize=8, fontweight="bold")
    ax.set(xlim=(0, 1), xlabel="macro-F1", title=title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BASE, label="classical baseline"), Patch(color=ATTN, label="neural model (attention pooling)"),
                       plt.Line2D([], [], marker="D", ls="", color="#222", label="recording-level")], loc="lower right", fontsize=8)
    return save(fig, folder, name)


def fold_spread(run_dir, models, folder=None):
    rows = []
    for m in models:
        s = read_json(Path(run_dir) / m / "summary.json")
        if s:
            rows += [{"model": nice(m), "macro-F1": f} for f in s["fold_f1"]]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(models) + 1.2))
    order = df.groupby("model")["macro-F1"].mean().sort_values().index
    sns.stripplot(data=df, y="model", x="macro-F1", order=order, ax=ax, color=ATTN, alpha=0.55, size=5, jitter=0.2)
    sns.pointplot(data=df, y="model", x="macro-F1", order=order, ax=ax, color="#222", errorbar="sd", markers="|", linestyles="none")
    ax.set(xlim=(0, 1), ylabel="", title="Every fold x seed (dots) and mean +- std (bar)")
    return save(fig, folder, "fold_spread")


def per_class(run_dir, models, classes, folder=None):
    M = [read_json(Path(run_dir) / m / "summary.json")["per_class_f1"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(models) + 1.4))
    sns.heatmap(np.array(M), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=[pretty(c) for c in classes],
                yticklabels=[nice(m) for m in models], ax=ax, cbar_kws={"label": "F1"})
    ax.set_title("F1 per class (pooled over folds and seeds)")
    ax.grid(False)
    return save(fig, folder, "per_class_f1")


def confusions(run_dir, models, classes, folder=None):
    n = len(models)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax, m in zip(axes.flat, models):
        cm = np.array(read_json(Path(run_dir) / m / "summary.json")["confusion"], float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, annot_kws={"fontsize": 7},
                    xticklabels=[c[:5] for c in classes], yticklabels=[c[:5] for c in classes])
        ax.set(title=nice(m), xlabel="predicted", ylabel="true")
        ax.grid(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return save(fig, folder, "confusion_matrices")


def confusion_single(run_dir, model, classes, folder=None, name=None):
    cm = np.array(read_json(Path(run_dir) / model / "summary.json")["confusion"], float)
    rowp = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    labels = np.array([[f"{p:.2f}\n({int(c)})" for p, c in zip(rp, rc)] for rp, rc in zip(rowp, cm)])
    sns.heatmap(rowp, annot=labels, fmt="",
                cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, annot_kws={"fontsize": 7},
                xticklabels=[pretty(c) for c in classes], yticklabels=[pretty(c) for c in classes])
    ax.set(title=f"{nice(model)}: confusion (row share, rep count)", xlabel="predicted", ylabel="true")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return save(fig, folder, name or f"confusion_{model}")


def deletion(table, folder=None):
    t = table[table["alpha_drop"].notna()].sort_values("alpha_drop")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.3))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["alpha_drop"], height=0.38, color=ATTN, label="top-10 % alpha frames masked")
    ax.barh(y - 0.2, t["random_drop"], height=0.38, color="#9aa4b1", label="10 % random frames masked")
    ax.set_yticks(y, t["name"])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set(xlabel="drop of the true-class probability", title="Does alpha point at the frames that matter?")
    ax.legend(fontsize=8, loc="lower right")
    return save(fig, folder, "alpha_deletion_test")


def alpha_profile(run_dir, model, classes, folder=None, bins=32):
    d = Path(run_dir) / model
    if not (d / "alpha.npy").exists():
        return None
    a = np.load(d / "alpha.npy")
    y = pd.read_csv(d / "preds.csv")["y"].to_numpy()
    prof = np.array([a[y == i].reshape(-1, bins, a.shape[1] // bins).sum(2).mean(0) if (y == i).any() else np.zeros(bins)
                     for i in range(len(classes))])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    sns.heatmap(prof * bins, cmap="Blues", ax=ax, xticklabels=False, yticklabels=[pretty(c) for c in classes],
                cbar_kws={"label": "alpha x bins (1 = uniform)"})
    ax.set(xlabel="rep start -> bottom -> end", title=f"Where {nice(model)} looks: mean alpha over the rep, per true class")
    ax.grid(False)
    return save(fig, folder, f"alpha_profile_{model}")


def learning_curves(run_dir, model, folder=None):
    d = Path(run_dir) / model
    hs = [(f.parent.name, read_json(f)) for f in sorted(d.glob("fold_0_*/history.json"))]
    if not hs:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, (fname, h) in enumerate(hs):
        ax[0].plot(h["train_loss"], color=ATTN, alpha=0.4 + 0.12 * i, label=f"fold {fname[-1]}")
        ax[1].plot(h["val_f1"], color=BASE, alpha=0.4 + 0.12 * i)
    ax[0].set(title=f"{nice(model)}: training loss (seed 0)", xlabel="epoch")
    ax[1].set(title="validation macro-F1 (validation persons)", xlabel="epoch", ylim=(0, 1))
    ax[0].legend(fontsize=7, ncol=5)
    return save(fig, folder, f"learning_curves_{model}")


def person_accuracy(run_dir, models, folder=None):
    rows = {}
    for m in models:
        p = pd.read_csv(Path(run_dir) / m / "preds.csv")
        rows[nice(m)] = p.assign(ok=p["y"] == p["pred"]).groupby("person")["ok"].mean()
    df = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(0.55 * df.shape[1] + 3, 0.42 * len(df) + 1.4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False, annot_kws={"fontsize": 7})
    ax.set(title="Accuracy per test person (the person is the main source of error)", xlabel="person", ylabel="")
    ax.grid(False)
    return save(fig, folder, "person_accuracy")


def ablation(table, folder=None, name="ablation"):
    t = table.copy()
    full = t.loc[t["variant"] == "full", "macro_f1"].iloc[0]
    t["delta"] = t["macro_f1"] - full
    t = t.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(t) + 1.2))
    cols = ["#9aa4b1" if v == "full" else (ATTN if d >= 0 else BASE) for v, d in zip(t["variant"], t["delta"])]
    ax.barh(t["variant"], t["delta"], xerr=t["macro_f1_std"], color=cols, height=0.6, error_kw=dict(ecolor="#666", capsize=3, lw=0.8))
    ax.axvline(0, color="#333", lw=0.8)
    for y, (d, f) in enumerate(zip(t["delta"], t["macro_f1"])):
        ax.text(d + (0.004 if d >= 0 else -0.004), y, f"{f:.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=8)
    ax.set(xlabel=f"change in macro-F1 vs the full model ({full:.3f}); bars: std over folds x seeds", title="Ablation")
    return save(fig, folder, name)
