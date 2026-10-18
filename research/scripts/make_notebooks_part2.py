# Notebooks 03 and 04. Executed at the end of make_notebooks.py (shares its helpers md, code, write, SETUP, PROTOCOL).
# flake8: noqa

# =============================================================================================== 03
write("03_ablation_best_model.ipynb", [
    md('''
# 03 · Ablation study of the best model (5-fold cross-validation by person)

The model chosen in notebook 02 is trained again with **one thing changed at a time**; data, folds, seeds and everything
else stay identical. The change in macro-F1 against the unchanged model (`full`) shows how much that part matters, and
the paired test on the same 15 runs says whether the change is larger than the noise.

| group | variants | question |
|---|---|---|
| A. inputs | joints only · angles only · no velocity · no tempo · lower-body joints only · no knee-offset / heel features | what information does the model use? |
| B. cameras | side camera only · front camera only (simulated: one body axis dropped) | is the two-camera setup needed? |
| C. preprocessing | camera-world axes instead of the body frame · metres instead of leg lengths · no rep trimming · T = 32 / 128 | does our preprocessing matter? |
| D. architecture | depends on the model: kernel / layers / dilation / receptive field / attention blocks; width 32 / 128; dropout 0.3 | which parts of the design are needed? |
| E. pooling | mean · max · mean + max instead of attention | does attention help, and what does α cost? |
| F. training | no noise augmentation · stronger noise · no label smoothing · no loss weighting · learning rate × 0.5 / × 2 | are the training choices justified? |
| G. data size | train on 33 % / 50 % / 75 % of the training persons | would more participants help? |
''' + PROTOCOL + '''
**Best configuration.** A change is adopted only if it raises the mean macro-F1 by ≥ 0.01 **and** wins on more of the
15 runs than it loses. Variants that remove α (mean / max pooling), simulate a missing camera or use less data are
reported but never adopted. Compatible improvements are then tested together once. The result is written to
`results/03_ablation/best_config.yaml` and used by notebook 04.

**How to run.** Run All after notebook 02. The expected time is printed below.  **Outputs.** `results/03_ablation/`.
'''),
    code(SETUP + '''
from squat_research.models import TRANSFORMERS
import yaml, seaborn as sns
OUT = results_dir(cfg, "03_ablation"); FIG = results_dir(cfg, "03_ablation", "figures"); RUN = OUT / "runs"
best_info = read_json(results_dir(cfg, "02_model_comparison") / "best_model.json")
best = best_info["model"]
print("ablating:", nice(best), f"(macro-F1 {best_info['macro_f1']:.3f} in notebook 02)")'''),
    code('''
V, GROUP = {"full": {}}, {"full": "reference"}
def add(group, **variants):
    for k, o in variants.items():
        V[k], GROUP[k] = o, group
add("A inputs", joints_only={"data.inputs": "joints"}, angles_only={"data.inputs": "frame"}, no_velocity={"data.velocity": False},
    no_tempo={"data.tempo": False}, lower_body_joints_only={"data.joints": "lower_body"},
    no_knee_heel_features={"data.drop_frame_features": ["knee_offset_L", "knee_offset_R", "heel_height_L", "heel_height_R"]})
add("B cameras", side_camera_only={"data.view": "side"}, front_camera_only={"data.view": "front"})
add("C preprocessing", world_axes={"data.body_frame": False}, metres={"data.leg_norm": False}, no_trim={"data.trim": False},
    T_32={"data.T": 32}, T_128={"data.T": 128})
ARCH = {
    "cnn1d": dict(kernel_3={"model.kernel": 3}, kernel_7={"model.kernel": 7}, kernel_9={"model.kernel": 9},
                  conv_layers_2={"model.conv_layers": 2}, conv_layers_4={"model.conv_layers": 4}, dilated={"model.dilated": True}),
    "tcn": dict(tcn_kernel_5={"model.tcn_kernel": 5}, tcn_levels_2={"model.tcn_levels": 2}, tcn_levels_3={"model.tcn_levels": 3},
                tcn_levels_5={"model.tcn_levels": 5}),
    "bilstm": dict(rnn_layers_1={"model.rnn_layers": 1}, rnn_layers_3={"model.rnn_layers": 3}),
    "bigru": dict(rnn_layers_1={"model.rnn_layers": 1}, rnn_layers_3={"model.rnn_layers": 3}),
    "cnn_lstm": dict(kernel_3={"model.kernel": 3}, kernel_7={"model.kernel": 7}, conv_layers_2={"model.conv_layers": 2},
                     rnn_layers_1={"model.rnn_layers": 1}),
    "st_transformer": dict(no_spatial_attention={"model.spatial_attn": False}, no_temporal_attention={"model.temporal_attn": False},
                           blocks_2={"model.n_blocks": 2}),
}
add("D architecture", **ARCH[best], width_32={"model.d_model": 32}, width_128={"model.d_model": 128}, dropout_0_3={"model.dropout": 0.3})
add("E pooling", mean_pooling={"model.pooling": "mean"}, max_pooling={"model.pooling": "max"}, mean_max_pooling={"model.pooling": "meanmax"})
add("F training", no_noise_aug={"train.noise_std": 0.0}, noise_0_03={"train.noise_std": 0.03}, no_label_smoothing={"train.label_smoothing": 0.0},
    no_loss_weighting={"train.weighting": False}, lr_0_0005={"train.lr": 0.0005}, lr_0_002={"train.lr": 0.002})
add("G data size", train_persons_33pct={"cv.train_person_frac": 0.33}, train_persons_50pct={"cv.train_person_frac": 0.5},
    train_persons_75pct={"cv.train_person_frac": 0.75})

est_h = best_info["minutes_per_15_runs"] * len(V) * 1.2 / 60
seeds_override = {}
if est_h > 6:                                     # very slow models: 1 seed per variant (still 5 person-wise folds)
    seeds_override, est_h = {"cv.seeds": [0]}, est_h / 3
V = {k: {**seeds_override, **o} for k, o in V.items()}
print(f"{len(V)} variants, about {est_h:.1f} h" + (" with 1 seed per variant (slow model)" if seeds_override else " with 3 seeds"))
pd.DataFrame({"variant": list(V), "group": [GROUP[k] for k in V], "change": [str({a: b for a, b in o.items() if a != "cv.seeds"}) for o in V.values()]})'''),
    md('## 1. Run the variants (background process + live monitor)'),
    code('''
st = read_json(RUN / "ablation_state.json")
if not st or st["state"] != "done" or not set(V) <= set(st["done"]):
    proc = monitor.launch_ablation(best, V, RUN)
try:
    while True:
        clear_output(wait=True); print(monitor.ablation_status(RUN, 4))
        st = read_json(RUN / "ablation_state.json") or {}
        pid = (RUN / "run.pid").read_text().strip() if (RUN / "run.pid").exists() else None
        if st.get("state") in ("done", "error") or (pid and not monitor._alive(pid) and st):
            break
        time.sleep(30)
except KeyboardInterrupt:
    print("stopped watching (the ablation continues)")'''),
    md('## 2. Results'),
    code('''
full = read_json(RUN / "full" / best / "summary.json"); full_scores = fold_scores(RUN / "full", best)
rows = []
for v in V:
    s = read_json(RUN / v / best / "summary.json")
    if not s:
        continue
    pt = paired_test(fold_scores(RUN / v, best), full_scores) if v != "full" else dict(mean_diff=0.0, wins=0, losses=0, p_value=1.0)
    rows.append(dict(variant=v, group=GROUP[v], macro_f1=s["macro_f1_mean"], macro_f1_std=s["macro_f1_std"], rec_macro_f1=s["rec_macro_f1_mean"],
                     delta=s["macro_f1_mean"] - full["macro_f1_mean"], wins=pt["wins"], losses=pt["losses"], p_value=pt["p_value"],
                     alpha_drop=s.get("alpha_drop"), median_best_epoch=float(np.median(s["best_epochs"])) if s["best_epochs"] else None))
abl = pd.DataFrame(rows).sort_values(["group", "delta"], ascending=[True, False]).reset_index(drop=True)
abl.to_csv(OUT / "ablation_table.csv", index=False)
(OUT / "ablation_table.md").write_text(abl.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
display(abl)'''),
    code('viz.ablation(abl, FIG, name=f"ablation_{best}"); plt.show()'),
    code('''
for g in sorted(set(abl["group"]) - {"reference"}):
    sub = pd.concat([abl[abl["variant"] == "full"], abl[abl["group"] == g]])
    viz.ablation(sub, FIG, name=f"ablation_{best}_{g.split()[0]}"); plt.title(f"{nice(best)} ablation, group {g}"); plt.show()'''),
    code('''
pc_full = np.array(full["per_class_f1"])
M = pd.DataFrame({v: np.array(read_json(RUN / v / best / "summary.json")["per_class_f1"]) - pc_full for v in abl["variant"] if v != "full"},
                 index=[c.replace("_", " ") for c in full["classes"]]).T
fig, ax = plt.subplots(figsize=(8, 0.33 * len(M) + 1.5))
sns.heatmap(M, annot=True, fmt="+.2f", cmap="RdBu", center=0, vmin=-0.4, vmax=0.4, ax=ax, annot_kws={"fontsize": 7})
ax.set_title(f"{nice(best)}: change in per-class F1 vs the full model"); viz.save(fig, FIG, f"ablation_per_class_{best}"); plt.show()'''),
    code('''
# learning curve over the number of training persons (group G)
lc = abl[abl["variant"].isin(["train_persons_33pct", "train_persons_50pct", "train_persons_75pct", "full"])].copy()
lc["fraction"] = lc["variant"].map({"train_persons_33pct": 0.33, "train_persons_50pct": 0.5, "train_persons_75pct": 0.75, "full": 1.0})
lc = lc.sort_values("fraction")
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.errorbar(lc["fraction"] * 12, lc["macro_f1"], yerr=lc["macro_f1_std"], marker="o", color=viz.ATTN, capsize=3)
ax.set(xlabel="training persons per fold (about)", ylabel="macro-F1", title="More participants -> better? (learning curve)", ylim=(0, 1))
viz.save(fig, FIG, f"learning_curve_persons_{best}"); plt.show()'''),
    md('## 3. Best configuration'),
    code('''
NOT_CANDIDATES = {"full", "mean_pooling", "max_pooling", "mean_max_pooling", "side_camera_only", "front_camera_only",
                  "train_persons_33pct", "train_persons_50pct", "train_persons_75pct"}
cand = abl[(~abl["variant"].isin(NOT_CANDIDATES)) & (abl["delta"] >= 0.01) & (abl["wins"] > abl["losses"])].sort_values("delta", ascending=False)
display(cand)
clean = lambda o: {k: x for k, x in o.items() if k != "cv.seeds"}
chosen, overrides = "full", {}
if len(cand):
    chosen, overrides = cand.iloc[0]["variant"], clean(V[cand.iloc[0]["variant"]])
    combo, keys = {}, set()
    for v in cand["variant"]:
        o = clean(V[v])
        if not keys & set(o):
            combo.update(o); keys |= set(o)
    if combo != overrides:
        print("testing the combination:", combo)
        V["combined"], GROUP["combined"] = {**seeds_override, **combo}, "best config"
        monitor.launch_ablation(best, {"combined": V["combined"]}, RUN).wait()
        sc = read_json(RUN / "combined" / best / "summary.json")
        print(f"combined: macro-F1 {sc['macro_f1_mean']:.3f} vs single best {cand.iloc[0]['macro_f1']:.3f}")
        if sc["macro_f1_mean"] > cand.iloc[0]["macro_f1"]:
            chosen, overrides = "combined", combo
print("best configuration:", chosen, overrides or "(defaults)")'''),
    code('''
best_cfg = apply_overrides(cfg, overrides)
s = read_json(RUN / chosen / best / "summary.json")
evidence = dict(model=best, name=nice(best), variant=chosen, overrides=overrides, macro_f1=s["macro_f1_mean"], macro_f1_std=s["macro_f1_std"],
                rec_macro_f1=s["rec_macro_f1_mean"], full_macro_f1=full["macro_f1_mean"], median_best_epoch=int(np.median(s["best_epochs"])),
                ablation_seeds=[0] if seeds_override else cfg["cv"]["seeds"])
best_cfg["final"] = dict(model=best, epochs=evidence["median_best_epoch"])
yaml.safe_dump(best_cfg, open(OUT / "best_config.yaml", "w"), sort_keys=False)
write_json(OUT / "best_config.json", evidence)
txt = (f"**Best configuration for {nice(best)}: `{chosen}`** {overrides if overrides else '(the default settings)'}: macro-F1 "
       f"{evidence['macro_f1']:.3f} ± {evidence['macro_f1_std']:.3f} (default {evidence['full_macro_f1']:.3f}). Final training length "
       f"{evidence['median_best_epoch']} epochs (median best epoch of the folds). Largest drops: "
       + ", ".join(f"{r.variant} ({r.delta:+.3f})" for r in abl.sort_values("delta").head(4).itertuples()) + ".")
(OUT / "summary.md").write_text(txt, encoding="utf-8"); display(Markdown(txt))'''),
])


# =============================================================================================== 04
write("04_final_model.ipynb", [
    md('''
# 04 · Final model: the best model with the best configuration

1. **Data** with the best configuration's preprocessing.
2. **Evidence**: the cross-validation results of notebooks 02 and 03 that led here (k = 5 by person, 3 seeds).
3. **Final training** on *all* persons for the median best epoch of the cross-validation folds (no held-out data is
   needed any more: the performance estimate comes from the cross-validation).
4. **Export**: `model.pt`, `config.json`, `norm_stats.json`, `model_card.md`.
5. **Inference** on preprocessed recordings: per rep the error, confidence, probabilities and α on the real frames,
   saved as `predictions.json` in the format the final product reads.
6. **Check on an unseen person** (`E_PANTS_MALE`, never used for training or model choice).

**How to run.** Run All after notebook 03 (a few minutes). **Outputs.** `results/04_final/`.
'''),
    code(SETUP + '''
import yaml
from squat_research.engine import train_final, export, predict_recording
from squat_research.common import path
from squat_research.dataset import dataset_file
OUT = results_dir(cfg, "04_final"); FIG = results_dir(cfg, "04_final", "figures"); PRED = results_dir(cfg, "04_final", "predictions")
best_cfg = yaml.safe_load(open(results_dir(cfg, "03_ablation") / "best_config.yaml", encoding="utf-8"))
evidence = read_json(results_dir(cfg, "03_ablation") / "best_config.json")
best, epochs = best_cfg["final"]["model"], best_cfg["final"]["epochs"]
print(f"final model: {nice(best)}, configuration '{evidence['variant']}' {evidence['overrides'] or ''}, {epochs} epochs")'''),
    md('## 1. Data'),
    code('''
if not dataset_file(best_cfg).exists():
    build_dataset(best_cfg)
ds = load_dataset(best_cfg); C = ds["classes"]
print(f"{len(ds['y'])} reps, {len(set(ds['recording']))} recordings, {len(set(ds['person']))} persons, T = {ds['joints'].shape[1]}")'''),
    md('## 2. Evidence (cross-validation, k = 5 by person, 3 seeds)'),
    code('''
for step, f in (("notebook 02, model choice", results_dir(cfg, "02_model_comparison") / "summary.md"),
                ("notebook 03, best configuration", results_dir(cfg, "03_ablation") / "summary.md")):
    display(Markdown(f"**{step}.** " + f.read_text(encoding="utf-8")))
cmp = pd.read_csv(results_dir(cfg, "02_model_comparison") / "comparison.csv")
display(cmp[["name", "group", "macro_f1", "macro_f1_std", "rec_macro_f1", "alpha_drop"]])'''),
    md('## 3. Final training on all persons'),
    code('''
t0 = time.time()
r = train_final(best, ds, best_cfg, epochs)
print(f"trained in {time.time() - t0:.0f} s, {r['n_params']:,} parameters")
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(1, len(r["history"]["train_loss"]) + 1), r["history"]["train_loss"], color=viz.ATTN, lw=2)
ax.set(xlabel="epoch", ylabel="training loss", title=f"{nice(best)}: final training on all persons")
viz.save(fig, FIG, "final_training_loss"); plt.show()'''),
    md('## 4. Export'),
    code('''
cv_epochs = read_json(results_dir(cfg, "03_ablation") / "runs" / evidence["variant"] / best / "summary.json")["best_epochs"]
EXP = export(r, best, ds, best_cfg, OUT / "export", extra=dict(cv_macro_f1=evidence["macro_f1"], cv_macro_f1_std=evidence["macro_f1_std"],
                                                            configuration=evidence["variant"], overrides=evidence["overrides"],
                                                            epochs_trained=epochs, cv_best_epochs=cv_epochs))
print(f"trained for {epochs} epochs = median of the best epochs of the 15 cross-validation runs {cv_epochs}")
card = f"""# Squat error model: {nice(best)}

* Classes: {", ".join(C)}
* Data: {len(ds['y'])} reps, {len(set(ds['recording']))} recordings, {len(set(ds['person']))} persons (two-camera Pose2Sim, 29.97 fps)
* Configuration: `{evidence['variant']}` {evidence['overrides'] or '(defaults)'}; trained on all persons for {epochs} epochs
* Cross-validation (k = 5 by person, 3 seeds): macro-F1 {evidence['macro_f1']:.3f} ± {evidence['macro_f1_std']:.3f}, recording level {evidence['rec_macro_f1']:.3f}
* Parameters: {r['n_params']:,}
* Output per rep: class probabilities and alpha (64 frame-attention weights) -> alpha on the real frames for the 3D body
* Files: model.pt (weights), config.json (model, data and feature settings), norm_stats.json (z-score statistics)
* Limitations: 18 persons; the person is the main source of error; abnormal_tempo is not modelled.
"""
(EXP / "model_card.md").write_text(card, encoding="utf-8"); print(sorted(p.name for p in EXP.iterdir())); display(Markdown(card))'''),
    md('''
## 5. Inference on a recording

`predict_recording(folder, export)` runs the whole chain on a preprocessed recording: features → normalisation → model →
per rep the error, confidence, probabilities and α; α is spread onto the real frames (10-frame windows, Gaussian
σ = 5 frames, peak 1) and multiplied by the confidence into an intensity timeline. These recordings were part of the
training data, so they show what the output looks like, not how accurate the model is.
'''),
    code('''
rec_root = path(best_cfg, "recordings")
man = pd.read_csv(path(best_cfg, "manifest"))
demo = man[man["include"]].groupby("label").head(1).head(3)
for row in demo.itertuples():
    p = predict_recording(rec_root / row.id, EXP)
    write_json(PRED / f"{row.id}.json", p)
    print(f"{row.id} (person {row.person}, label {row.label}):", pd.Series([x["label"] for x in p["reps"]]).value_counts().to_dict())
    n = len(p["reps"])
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.2), sharey="row", squeeze=False)
    for i, rep in enumerate(p["reps"]):
        axes[0, i].plot(np.linspace(0, 100, len(rep["flex"])), rep["flex"], color="#333")
        axes[0, i].set_title(f"rep {rep['rep']}: {rep['label'].replace('_', ' ')}\\n{rep['confidence']:.2f}", fontsize=7)
        axes[1, i].imshow(np.array(rep["alpha_frames"])[None, :], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[1, i].set_yticks([]); axes[1, i].set_xticks([])
    axes[0, 0].set_ylabel("knee flexion (deg)"); axes[1, 0].set_ylabel("alpha")
    fig.suptitle(f"{row.id} ({row.label}): knee flexion per rep and alpha on the real frames", fontweight="bold")
    fig.tight_layout(); viz.save(fig, FIG, f"inference_{row.id}"); plt.show()'''),
    md('## 6. Check on a person the model has never seen (`E_PANTS_MALE`)'),
    code('''
unseen = man[(man["person"] == "E_PANTS_MALE") & (man["label"].isin(C))]
rows = []
for row in unseen.itertuples():
    try:
        p = predict_recording(rec_root / row.id, EXP)
    except Exception as e:
        print(row.id, "skipped:", e); continue
    write_json(PRED / f"{row.id}.json", p)
    labels = [x["label"] for x in p["reps"]]
    vote = int(pd.DataFrame([x["probs"] for x in p["reps"]]).mean().idxmax()) if p["reps"] else None
    rows.append(dict(recording=row.id, true=row.label, reps=len(labels), rep_accuracy=np.mean([l == row.label for l in labels]) if labels else None,
                     recording_vote=C[vote] if vote is not None else None))
unseen_tab = pd.DataFrame(rows); unseen_tab.to_csv(OUT / "unseen_person_check.csv", index=False); display(unseen_tab)
if len(unseen_tab):
    print(f"rep accuracy on the unseen person: {unseen_tab['rep_accuracy'].mean():.2f}; recordings right: "
          f"{(unseen_tab['true'] == unseen_tab['recording_vote']).sum()}/{len(unseen_tab)} (left out of training for poor pose quality)")'''),
    md('''
## 7. Using the export

```python
from squat_research.engine import predict_recording
pred = predict_recording(r"<preprocessed recording folder>", r"research_folders/results/04_final/export")
```
`pred` has the fields of the final product's `predictions.json` (`classes`, `fps`, `n_frames`, per rep `label`, `confidence`,
`probs`, `alpha`, `alpha_frames`, and `intensity` over the whole timeline).
'''),
])
