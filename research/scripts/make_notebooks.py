"""make_notebooks.py - writes the four research notebooks into research_folders/notebooks/ (re-run after editing)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
KERNEL = {"display_name": "Python (SquatModelVenv)", "language": "python", "name": "squatmodel"}

SETUP = '''# ---- setup (every notebook starts with this cell) ----
import sys, json, time, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()     # research_folders/
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown, clear_output, Image
pd.set_option("display.precision", 3); pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 120)
from squat_research.common import load_config, apply_overrides, results_dir, nice, read_json, write_json
from squat_research.dataset import build_dataset, load_dataset
from squat_research.engine import comparison_table, select_best, fold_scores, paired_test
from squat_research import viz, monitor
cfg = load_config()
print("research folder:", ROOT)'''

PROTOCOL = '''
**Evaluation protocol (identical for every model and every variant).**
* **5-fold cross-validation grouped by person** (StratifiedGroupKFold, k = 5): each fold tests 3-4 people the model has
  never seen; every person is tested once per seed.
* **3 seeds** (0, 1, 2): 15 runs per model; the reported number is the mean ± standard deviation over the 15 runs.
* **Validation persons**: inside each fold, 2 training persons choose the best epoch;
  the test persons are never used for any choice.
* **Early stopping**: training stops when the validation macro-F1 has not improved for **20 epochs** (at most 60 epochs);
  the weights of the best epoch are kept and saved as a checkpoint (`<model>/fold_<seed>_<k>/best.pt`, with the best epoch,
  its validation F1, the number of epochs run, the normalisation statistics and the test metrics).
* **Normalisation** statistics from the training part of each fold only; **loss weights** balance the classes and let
  every recording count equally.
* **Metrics**: macro-F1 per rep (main), accuracy, macro-F1 per recording (mean probability over its reps), per-class F1,
  confusion matrix; **paired Wilcoxon test** on the 15 identical runs when two models or variants are compared.
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    NB.mkdir(exist_ok=True)
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


WATCH = '''
if not (RUN / "progress.json").exists() or read_json(RUN / "progress.json")["state"] != "done":
    proc = monitor.launch(MODELS, RUN, save_models=True)     # checkpoints: <model>/fold_<seed>_<k>/best.pt (best epoch)
monitor.watch(RUN, every=30)      # interrupt to stop watching; the training continues (monitor.stop(RUN) stops it)'''

# =============================================================================================== 01
write("01_preprocessing_and_dataset.ipynb", [
    md('''
# 01 · Preprocessing and dataset

**Task.** Classify every squat repetition into one of 6 classes (`correct`, `insufficient_depth`, `forward_lean`,
`knee_valgus`, `knee_varus`, `heel_lift`) from the 3D skeleton of the two-camera (front + side) Pose2Sim pipeline.

This notebook builds the dataset, **checks that the preprocessing is correct**, and visualises the data.
No model is trained here.

**How to run.** Kernel *Python (SquatModelVenv)*, Run All (about 1 minute).
**Outputs.** `results/01_dataset/` (tables and figures).
'''),
    code(SETUP + '''
OUT = results_dir(cfg, "01_dataset")'''),
    md('''
## 1. Which recordings are used

| rule | why |
|---|---|
| class `abnormal_tempo` left out | people did not follow the speed protocol; the error is a rhythm across reps, not visible in one rep |
| person `E_PANTS_MALE` left out of training | loose trousers: neck, shoulders and legs poorly triangulated (used as an unseen-person check in notebook 04) |
| `p01` left out | early test export of p1, not in KEY_INFO |
| KEY_INFO file errors (p4, p50, p70, p71, p76, p105) | no output or only 0-2 reps recorded |
| rep filters | lower-body joint missing > 25 % of a rep; Neck missing > 50 % (the pipeline's trunk angles are wrong without it); duration outside 0.5-10 s after trimming |
'''),
    code('''
man, npz = build_dataset(cfg)
display(man[~man["include"]][["id", "person", "label", "reason"]].reset_index(drop=True))
print(f"{man['include'].sum()} of {len(man)} recordings included")
ds = load_dataset(cfg); C = ds["classes"]
summary = pd.DataFrame({"recordings": [len(set(ds["recording"][ds["y"] == i])) for i in range(len(C))],
                        "reps": np.bincount(ds["y"], minlength=len(C)),
                        "persons": [len(set(ds["person"][ds["y"] == i])) for i in range(len(C))]}, index=C)
summary.loc["total"] = [len(set(ds["recording"])), len(ds["y"]), len(set(ds["person"]))]
summary.to_csv(OUT / "dataset_summary.csv")
print(f"{len(ds['y'])} reps used, {ds['n_filtered']} reps removed by the quality filters"); display(summary)'''),
    md('''
## 2. Preprocessing (per recording, `squat_research/features.py`)

| step | what | why |
|---|---|---|
| 1. load | `points3d.trc`: 22 joints, metres, 29.97 fps. Joints written as exactly (0,0,0) are **missing** and interpolated in time; `Hip` is rebuilt from the two hip joints | Pose2Sim writes untriangulated joints as (0,0,0) |
| 2. body frame | *up* = real standing direction ankles → hips (frames where the hips are highest); *forward* = heel → toe of both feet; *lateral* = up × forward; left / right labels checked | the camera-world axes depend on how the calibration board was held; the Neck is missing for whole recordings of some people, so it is not used |
| 3. normalise | hip-centred, divided by the leg length (hip-knee + knee-ankle) | removes body size and position |
| 4. cut + trim | reps from `reps.csv`, trimmed to the movement: start / end where knee flexion is within 10 % of its range above the standing level, **taken separately before and after the bottom** | people rest between reps with slightly bent knees; that rest must not be part of the rep |
| 5. resample | each rep to T = 64 frames (linear) | equal length for every model |
| 6. inputs | joint tokens 64 × 22 × 6 (position + velocity), frame features 64 × 19 (14 pipeline angles + knee offset L/R, heel height L/R, hip height), tempo 4 (duration, descent, ascent, bottom fraction), summary row 46 (classical baselines) | resampling hides speed, so tempo is kept separately |
'''),
    md('## 3. Is the preprocessing correct? (checks on every rep)'),
    code('''
from squat_research.checks import run_checks
chk = run_checks(ds); chk.to_csv(OUT / "preprocessing_checks.csv", index=False)
print(f"{(chk['result'] == 'PASS').sum()} of {len(chk)} checks pass")
display(chk)'''),
    code('''
# trimming examples: knee flexion of the whole recording (grey) and the trimmed reps used by the models (blue)
from squat_research.features import load_recording, trim_rep, ANGLE_COLS
from squat_research.common import path
rec_ids = list(dict.fromkeys(ds["recording"]))[:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
for ax, rid in zip(axes, rec_ids):
    rec = load_recording(path(cfg, "recordings") / rid)
    a = rec["angles"]; flex = np.nanmean(a[:, [ANGLE_COLS.index("knee_flexion_L"), ANGLE_COLS.index("knee_flexion_R")]], axis=1)
    ax.plot(flex, color="#bbb", lw=1)
    for _, r in rec["reps"].iterrows():
        s, e, b = trim_rep(flex, int(r["start_frame"]), min(int(r["end_frame"]), len(flex) - 1))
        ax.plot(range(s, e + 1), flex[s:e + 1], color=viz.ATTN, lw=1.6); ax.axvline(r["start_frame"], color="#ddd", lw=0.6)
    ax.set(title=f"{rid}: trimmed reps (grey lines = reps.csv boundaries)", xlabel="frame", ylabel="knee flexion (deg)")
viz.save(fig, OUT, "trimming_examples"); plt.show()'''),
    md('## 4. The dataset'),
    code('viz.class_counts(ds, OUT); plt.show()'),
    code('viz.person_class_matrix(ds, OUT); plt.show()'),
    code('viz.skeleton_example(ds, OUT); plt.show()'),
    code('viz.flex_curves(ds, OUT); plt.show()'),
    code('''
viz.feature_boxplots(ds, OUT); plt.show()
viz.rep_features_table(ds).groupby("class").median().to_csv(OUT / "per_class_feature_medians.csv")'''),
    md('''
Each error has a visible signature: knee offset inward for valgus and outward for varus, heel height for heel lift,
less knee flexion and hip drop for insufficient depth, more trunk flexion for forward lean. `correct` overlaps with
several classes because each person's normal squat is different, so the person is the main source of variation and
every split keeps all reps of a person on one side.
'''),
    code('''
viz.quality(ds, OUT); plt.show()
q = pd.DataFrame(ds["meta"]["recordings"]).T[["leg_length_m", "up_tilt_deg", "n_reps"]].astype(float)
q.to_csv(OUT / "recording_quality.csv"); display(q.describe().T)'''),
])

# =============================================================================================== 02
write("02_model_comparison.ipynb", [
    md('''
# 02 · Model comparison (5-fold cross-validation by person)

| group | models |
|---|---|
| classical baselines | logistic regression, random forest, XGBoost (on the 46-number summary row + tempo) |
| neural models (attention pooling → α) | 1D-CNN, BiLSTM, BiGRU, CNN-LSTM, TCN, ST-Transformer |

Every neural model ends with **attention pooling over the 64 frames**: its weights α say which moments decided the class.
The final product uses α to light up the error region on the 3D body at the right time, so α is checked too.
''' + PROTOCOL + '''
**How to run.** Run All after notebook 01. About 1 hour on the laptop RTX 3050 (the ST-Transformer takes ~50 min of it).
The monitor cell shows progress and ETA; interrupting it does not stop the training.

**Outputs.** `results/02_model_comparison/` (comparison tables, figures, one report per model, `best_model.json`).
'''),
    code(SETUP + '''
from squat_research.models import MODEL_INFO, build_model, n_params, TREE_MODELS
from squat_research.report import model_report
OUT = results_dir(cfg, "02_model_comparison"); FIG = results_dir(cfg, "02_model_comparison", "figures")
REP = results_dir(cfg, "02_model_comparison", "model_reports"); RUN = OUT / "cv"
ds = load_dataset(cfg); C = ds["classes"]
MODELS = cfg["baseline_models"] + cfg["neural_models"]'''),
    md('## 1. The models: what each receives and how it processes it'),
    code('''
T, J, F = ds["joints"].shape[1], ds["joints"].shape[2], ds["frame"].shape[2]
models_tab = pd.DataFrame([{"model": nice(m), "family": MODEL_INFO[m]["family"], "input": MODEL_INFO[m]["input"],
                            "processing": MODEL_INFO[m]["processing"],
                            "parameters": "-" if m in TREE_MODELS else f"{n_params(build_model(m, J, F, T, len(C), ds['meta']['joint_names'], cfg['model'])):,}"}
                           for m in MODELS])
models_tab.to_csv(OUT / "models_overview.csv", index=False); display(models_tab)'''),
    md('## 2. Train all models (background process + live monitor)'),
    code(WATCH),
    md('## 3. Results'),
    code('''
table = comparison_table(RUN)
table["group"] = ["classical baseline" if m in TREE_MODELS else "neural" for m in table["model"]]
table.to_csv(OUT / "comparison.csv", index=False)
(OUT / "comparison.md").write_text(table.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
display(table[["name", "group", "macro_f1", "macro_f1_std", "accuracy", "rec_macro_f1", "alpha_drop", "random_drop", "params", "median_best_epoch", "minutes"]])'''),
    code('viz.comparison(table, FIG, title="Macro-F1 per rep (5-fold cross-validation by person x 3 seeds)"); plt.show()'),
    code('viz.fold_spread(RUN, list(table["model"]), FIG); plt.show()'),
    code('viz.per_class(RUN, list(table["model"]), C, FIG); plt.show()'),
    code('viz.confusions(RUN, list(table["model"]), C, FIG); plt.show()'),
    code('viz.person_accuracy(RUN, list(table["model"]), FIG); plt.show()'),
    md('''
## 4. Does α point at the frames that matter?

For every test rep, the 10 % of frames with the highest α are replaced by the rep's mean and the drop of the true-class
probability is measured; the same with 10 % random frames (3 repeats). A meaningful α hurts clearly more when masked.
'''),
    code('viz.deletion(table[table["alpha_drop"].notna()], FIG); plt.show()'),
    md('## 5. One report per model (input processing and performance)'),
    code('''
for m in table["model"]:
    display(Markdown(model_report(RUN, m, ds, cfg, REP)))
    for f in ["confusion.png", f"learning_curves_{m}.png", f"alpha_profile_{m}.png"]:
        if (REP / m / f).exists():
            display(Image(filename=str(REP / m / f), width=640))'''),
    md('''
## 6. Choosing the best model

**Rule** (fixed before looking at the numbers):
1. Candidates: neural models whose α passes the check (masking the α frames drops the true-class probability ≥ 2 × more than random frames).
2. Highest mean macro-F1 over the 15 runs; models within one standard deviation of it count as tied.
3. Among tied models: higher recording-level macro-F1, then fewer parameters (the product runs on a CPU).

The paired test compares the chosen model with every other model on the same 15 runs (p < 0.05 = a real difference).
'''),
    code('''
neural = table[table["group"] == "neural"].reset_index(drop=True)
best, rule = select_best(neural)
display(rule[["name", "macro_f1", "macro_f1_std", "rec_macro_f1", "alpha_drop", "random_drop", "alpha_ok", "tied_with_best", "params", "chosen"]])
fb = fold_scores(RUN, best)
paired = pd.DataFrame([{"compared with": r["name"], **paired_test(fb, fold_scores(RUN, r["model"]))} for _, r in table.iterrows() if r["model"] != best])
paired.to_csv(OUT / "paired_tests_best_vs_others.csv", index=False); display(paired)'''),
    code('''
b = table.set_index("model").loc[best]
info = dict(model=best, name=nice(best), macro_f1=b["macro_f1"], macro_f1_std=b["macro_f1_std"], rec_macro_f1=b["rec_macro_f1"],
            alpha_drop=b["alpha_drop"], random_drop=b["random_drop"], params=int(b["params"]), median_best_epoch=b["median_best_epoch"],
            tied=list(rule.loc[rule["tied_with_best"], "model"]), minutes_per_15_runs=b["minutes"])
write_json(OUT / "best_model.json", info)
cls = table[table["group"] == "classical baseline"].iloc[0]
txt = (f"**Chosen model: {info['name']}**: macro-F1 {info['macro_f1']:.3f} ± {info['macro_f1_std']:.3f}, recording level "
       f"{info['rec_macro_f1']:.3f}, α check {info['alpha_drop']:.3f} vs random {info['random_drop']:.3f}, {info['params']:,} parameters. "
       f"Tied within one standard deviation: {', '.join(nice(m) for m in info['tied'])}. Best classical baseline: {cls['name']} ({cls['macro_f1']:.3f}).")
(OUT / "summary.md").write_text(txt, encoding="utf-8"); display(Markdown(txt))'''),
])

exec(open(Path(__file__).with_name("make_notebooks_part2.py"), encoding="utf-8").read())
