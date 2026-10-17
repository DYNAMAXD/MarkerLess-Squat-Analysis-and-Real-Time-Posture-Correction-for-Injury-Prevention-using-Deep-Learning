# Squat error study: workflow, models, ablation, and how to compare (written 2026-10-17)

Audience: a teammate (and the Claude Code instance on their laptop) who wants to understand this
workflow, compare it with their own model, and run it. Everything here was checked against the code in
`squat_error_study/`. When this document and the code disagree, the code wins.

## 0. Prompt to give Claude on the other laptop

> Read `squat_error_study/docs/WORKFLOW_FOR_COMPARISON.md` and `squat_error_study/README.md`.
> Then (1) set up the environment as in section 7, (2) run the 4 baseline models, (3) compare my model
> (describe or point to it) with this workflow using section 8. Use the SAME person-wise folds,
> the SAME 6 classes and the SAME metrics. Report differences in input processing, splits and
> metrics before comparing any numbers. Do not change files under `squat_error_study/src` without telling me.

## 1. The task

Classify every squatting **repetition** into one of **6 classes**:
`correct, insufficient_depth, forward_lean, knee_valgus, knee_varus, heel_lift`.
Input is 3D skeleton data from two ordinary cameras (front + side) processed with Pose2Sim
(`points3d.trc`, frame-wise angles `advancedangles.csv`, rep boundaries `reps.csv`, per-rep summary
`rep_features.csv`, `results.json`).
`abnormal_tempo` is **removed from the study** (people did not follow the speed protocol, so most reps were normal speed).
The recordings with that label are excluded in `build_manifest.py` (`DROPPED_LABEL`). The word "tempo" still appears in
the code as 4 **input numbers** (rep duration etc.), which are a feature, not a class.

## 2. Data decisions (what is in, what is out)

State after the data was re-copied from the original on 2026-10-17:

| Item | Number |
|---|---|
| recordings found (p1-p133, no p122 exists in KEY_INFO or the mapping files) | 132 |
| included | **103 recordings, 18 persons (A-S without E), 1,068 reps** |
| reps left after quality filters (Neck-missing reps removed) | **933 reps, 91 recordings** |
| reps per class after filters (correct, depth, lean, valgus, varus, heel) | 134, 133, 170, 144, 163, 189 |

Excluded and why (`data/manifest.csv`, column `exclude_reason`):

* person `E_PANTS_MALE` (7 recordings, asked to ignore)
* `abnormal_tempo` recordings (class dropped)
* KEY_INFO errors: **p4** (no reps.csv), **p50, p70, p71, p76** (1-2 reps), **p105** (0 reps)

### KEY_INFO findings (what is wrong and how it can be solved)

| Recording | KEY_INFO says | What the files/logs show | Fix |
|---|---|---|---|
| p4 (A, knee_valgus) | no csv reports | `batch_log.txt`: `RuntimeError: Only 1.1 s of squat video after the clap in one of the cameras`. Only calibration/board files exist | Cannot be fixed from processed data. Re-record, or re-run Pose2Sim on the raw videos with a different clap/start setting. Until then excluded |
| p50, p70, p71, p76 | just 1-2 reps recorded | Processing took ~18 s (normal ~65 s), so the videos are short: the person did only 1-2 squats | Cannot create reps. Excluded. If the raw video is longer than what was processed, re-run with the clap window changed |
| p105 (P, correct) | no rep data | `reps.csv` has only a header | Same as above. Excluded |
| p93 (N) | not flagged | Was missing `reps.csv` in an earlier copy; the full re-copy fixed it (10 reps) | Included now |
| p122 | absent | No ID in KEY_INFO or mapping files | Probably never recorded. Nothing to fix |
| p42, p87, p92, p95 | not flagged | `batch_log.txt` prints "error ... [stage] done" but the folders are complete | Harmless log wording. Included |

Other recording-quality notes: `summary.csv` (batch p119-p133) marks p126, p127, p131, p132 as "clap sync uncertain"; they are
kept. Neck/Head/shoulders were not triangulated for whole recordings of persons B, D, F (and a few more):
those reps (135 of 1,068) are dropped by `data.max_trunk_missing: 0.5`. This removes more reps from `correct` and `insufficient_depth`,
so those two classes are the smallest after filtering. Mention it in the paper's limitations. Per-recording quality: `docs/data_quality_report.html`.

## 3. Data processing (shared by all models) - `src/squat_study/features.py`

1. **Load** `points3d.trc` (metres). Pose2Sim writes joints it could not triangulate as exactly (0,0,0): these are set to NaN
   (missing) and linearly interpolated in time. If `Hip` is missing but both hip joints exist, `Hip` = their midpoint.
2. **Body frame.** For each recording the axes are derived from the person, not the camera: *up* = mean(hips) - mean(ankles),
   *forward* = mean(toe - heel) of both feet, *lateral* = up x forward. Joints are hip-centred and divided by the
   **leg length** (median hip-knee-ankle length), so different body sizes and camera setups become comparable.
   If the pipeline's left/right labels are swapped for a recording, they are swapped back.
3. **Per rep.** Cut `[start_frame, end_frame]` from `reps.csv`. Drop reps shorter than 1 s or longer than 12 s.
   Compute joint **velocity** (leg lengths per second). Build 19 frame features: 14 pipeline angles + 5 extras
   (`knee_valgus_body_L/R`, `heel_height_L/R`, `hip_height_rel`).
4. **Resample** each rep to **T = 64 frames** (linear interpolation) so all reps have equal length.
5. **Tempo numbers** (4): rep duration, descent time, ascent time, bottom fraction. Kept separately because resampling hides speed.
6. **Summary row**: the rep's row of `rep_features.csv` (46 numbers: min/max/mean-type statistics per angle).
7. Saved in `data/processed/reps_T64.npz` (+ `.json`) by `scripts/build_dataset.py`.

Resulting arrays: `joints [N, 64, 22, 6]` (22 joints x [pos lat/up/fwd, vel lat/up/fwd]), `frame [N, 64, 19]`,
`tempo [N, 4]`, `summary [N, 46]`, plus `label, person, recording, rep_no, missing, trunk_missing`.

**Splits (data.py):** `StratifiedGroupKFold`, 5 folds, grouped by **person**, so one person is never in train and test.
Two training persons are held out as a validation set to choose the best epoch and for early stopping (patience 15);
the test fold is never used for selection.
**Normalisation (data.py `Normaliser`):** per-channel z-score with mean/std from the training part of each fold only.
**Weights:** classes balanced; each recording counts equally whatever its number of reps.
**Metrics:** macro-F1 and accuracy per rep, plus a recording-level vote (mean probabilities over the reps of a recording).
Mean +- std over folds (and seeds).

## 4. What each model receives - `train.py`, `models.py`

| Model | Input | Processing / architecture | alpha (frame attention) |
|---|---|---|---|
| `decision_tree` | 46 summary + 4 tempo = 50 numbers | max_depth 4, sample weights by recording | no |
| `logreg` | same 50 | median impute, StandardScaler, LogisticRegression (C=1) | no |
| `random_forest` | same 50 | median impute, 300 trees | no |
| `xgboost` | same 50 | median impute, 300 trees, depth 4, lr 0.1 | no |
| `cnn1d` | per frame: 22x6 joints flattened + 19 features = 151 channels, T=64 | 3 Conv1d layers (kernel 5, BatchNorm, GELU), mean+max pooling over time, + tempo into the head | no |
| `bigru_attn` | same 151 channels | Linear projection, 2-layer bidirectional GRU, **attention pooling**, + tempo | yes |
| `bilstm` | same 151 channels | 2-layer bidirectional LSTM, **mean pooling**, + tempo | no |
| `stgcn` | joints as a graph: `[6 channels, T, 22 joints]`, bones from `BONES` (Hip-RHip-RKnee-..., Neck-Head, arms) | 3 spatio-temporal graph-conv blocks, global mean; the 19 frame features go through a small MLP and are concatenated, + tempo | no |
| `transformer` (proposed, from the lead's handoff) | 22 joint tokens (6 values each) + 1 frame token (19 features) per frame | each token embedded to d=64 plus a joint-id and a time embedding; 4 blocks of **spatial attention** (over the 23 tokens of a frame) then **temporal attention** (over the 64 frames, per token) then MLP; the T frame tokens go to **attention pooling**; + tempo into the head | yes |

All neural models: AdamW (lr 1e-3, weight decay 0.01), cosine schedule, 60 epochs max, batch 64, label smoothing 0.05,
Gaussian noise 0.01 on joint channels (training only), gradient clipping 1.0. All settings are in `configs/default.yaml`.

## 5. The ablation study

**Idea:** train the same model again with **one thing removed or changed**, keep everything else (data, folds, seed, training)
identical, and see how much the macro-F1 drops. A big drop means that part matters; no drop means it is not needed.
The reference row is `full`.

**Code:** `scripts/run_ablation.py`. A dictionary `VARIANTS` maps a variant name to config overrides (`"section.key": value`).
For each variant it deep-copies the config, applies the overrides, loads the right dataset file, and calls the same
`train.run_experiment` used for the main comparison. `data.apply_switches` zeroes input parts after z-scoring
(zero = the feature carries no information). Adding an ablation = adding one line to `VARIANTS`.

| Variant | What changes | Question answered |
|---|---|---|
| `full` | nothing | reference |
| `joints_only` | angle features zeroed | do the joints alone suffice? |
| `angles_only` | joints zeroed | do the angles alone suffice? |
| `no_velocity` | velocity channels zeroed | does motion speed help? |
| `no_tempo` | the 4 tempo numbers zeroed | does rep timing help? |
| `view_side_only` | drops the lateral axis (+ angle features) | what if only a side camera existed? (**simulated**, not a real single-camera run) |
| `view_front_only` | drops the forward axis | same for a front camera |
| `no_spatial_attn` | spatial attention replaced by mean mixing | does attention between joints help? |
| `no_temporal_attn` | temporal attention removed | does attention over time help? |
| `mean_pooling` | attention pooling replaced by mean | does the pooling attention help (and costs alpha)? |
| `blocks_2` | 2 transformer blocks instead of 4 | is depth needed? |
| `no_noise_aug` | noise augmentation off | does augmentation help? |
| `T32`, `T128` | resample length 32 / 128 | how sensitive to T? (needs `build_dataset.py --T 32` and `--T 128` first) |

Run: `python scripts\run_ablation.py` (all), `--variants full no_velocity` (some), `--model bigru_attn` (other model),
`--epochs 3 --splits 3` (quick check). Output: `results/runs/<time>_ablation_<model>/` with tables and figures.
Use 3 seeds (`--seeds 0 1 2`) for numbers that go into the paper; with 18 persons the fold-to-fold std is large (about 0.05), so
only differences clearly larger than that are meaningful.

## 6. Current numbers (cheap baselines only, seed 0, 5 folds, 6 classes, 933 reps)

| Model | Macro-F1 (rep) | Macro-F1 (recording vote) |
|---|---|---|
| decision_tree | 0.375 +- 0.060 | 0.404 |
| logreg | 0.481 +- 0.074 | 0.498 |
| random_forest | 0.491 +- 0.031 | 0.497 |
| xgboost | 0.539 +- 0.053 | 0.571 |

The neural models and the ablation still have to be run (GPU). Chance level for 6 classes is about 0.17.

## 7. Setting up on another laptop

```powershell
git clone <repo-url>
cd <repo>
git checkout squat-error-study                       # the branch with squat_error_study/
cd squat_error_study
# Windows + NVIDIA GPU:   .\scripts\setup_env.ps1 -Gpu      (CPU only: .\scripts\setup_env.ps1)
# Linux + NVIDIA GPU:     bash scripts/setup_env.sh gpu
# the script creates a virtual environment and prints its python path; use that path below as <PY>
<PY> -c "import torch; print(torch.cuda.is_available())"   # True on the GPU machine
<PY> scripts\run_cv.py --models decision_tree logreg random_forest xgboost --name baselines   # about 1 minute, checks the setup
<PY> scripts\run_cv.py --name main --seeds 0 1 2                                                # all 9 models
<PY> scripts\build_dataset.py --T 32 ; <PY> scripts\build_dataset.py --T 128                    # only for the T variants
<PY> scripts\run_ablation.py --seeds 0 1 2
```

* `data/processed/reps_T64.npz` is in git, so nothing has to be rebuilt. `data/recordings/` is not (git-ignored). It is only
  needed to rebuild (`build_manifest.py` reads the original `Temp_Dataset`; the T32/T128 builds need it).
  If the laptop has no `data/recordings/`, copy it by hand or run the T variants on the machine that has it.
* Model weights (`*.pt`, `*.joblib`) are git-ignored; tables, figures and json of each run are tracked.
* Long path warning: on Windows keep the venv at a short path (for example `C:\squat_venv`); a long repo path can break the torch install.
* Rough time: transformer ~77 s per epoch on a CPU, minutes on a GPU.

## 8. How to compare with your friend's model

Check these **before** comparing numbers, because each one can change the score by more than the model does:

1. **Split.** Person-wise (this study) or by recording/rep? A split by rep puts the same person in train and test and gives much higher scores (leakage).
2. **Classes.** 6 classes here. Is `abnormal_tempo` in? Different class sets cannot be compared.
3. **Which recordings/reps.** Same exclusions (E person, KEY_INFO errors, Neck-missing reps)? Check counts: 103 recordings / 1,068 reps / 933 after filters.
4. **Input.** Raw camera coordinates or body frame normalised by leg length? Frames resampled to what length? Angles from the pipeline or recomputed?
5. **Model selection.** Is the best epoch chosen on a separate validation set (here: 2 held-out training persons) or on the test fold?
6. **Metric.** Macro-F1 vs accuracy, per rep vs per recording.
7. **Variance.** Mean +- std over folds and seeds; with 18 persons, differences under about 0.05 are noise.

Then add the friend's model: write a class with the signature `logits, alpha = model(joints[B,T,22,6], frame[B,T,19], tempo[B,4])`,
register it in `models.build_model` and `NEURAL_MODELS`, and run `run_cv.py --models <name>`. It then uses the identical folds,
normalisation, weights and report. Put its results in a separate run folder and compare the `tables/` of both runs.

## 9. Files to know

`README.md` (commands), `configs/default.yaml` (all settings), `data/manifest.csv` (who is in/out and why),
`src/squat_study/{features,data,models,train,report}.py`, `scripts/{build_manifest,build_dataset,run_cv,run_ablation,make_data_report}.py`,
`docs/data_quality_report.html`, `docs/session_report_2026-10-17.html` (older, from before the data re-copy: its counts are outdated).
