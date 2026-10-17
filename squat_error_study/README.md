# squat_error_study

Person-wise comparison of squat-error classifiers (per rep, 6 classes) on the two-camera Pose2Sim dataset,
with ablation study and figures for the review and the paper.

All commands are run from this folder (`squat_error_study/`). `C:\squat_venv` is the Python environment.

> **Start here for the full workflow, model inputs, ablation and comparison guide: `docs/WORKFLOW_FOR_COMPARISON.md`.**
> Data state after the 2026-10-17 re-copy: 103 recordings, 1,068 reps (933 after quality filters), 18 persons, 6 classes. p94-p99 are restored (Step 0 below is done).

## 1. Environment

| Machine | Command |
|---|---|
| This PC (CPU) | already created at `C:\squat_venv` |
| Lab PC, Windows, NVIDIA GPU | `git pull`, then `.\scripts\setup_env.ps1 -Gpu` |
| Lab PC, Linux, NVIDIA GPU | `git pull`, then `bash scripts/setup_env.sh gpu` |

The venv is never in git. `requirements.txt` (and the exact versions in `requirements.lock.txt`) recreate it.
The notebooks need two more packages, added to `requirements.txt` after the first install:

```powershell
C:\squat_venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Folder map

| Path | What |
|---|---|
| `configs/default.yaml` | every setting (classes, folds, epochs, switches). Edit or copy it, never edit code for a setting |
| `data/manifest.csv` | one row per recording: person, label, reps, include / exclude + reason |
| `data/recordings/` | the 103 copied recordings (6 classes) (**git-ignored**; copy by hand to the lab if you need to rebuild) |
| `data/processed/reps_T64.npz` (+ `.json`) | the compact dataset every model trains from (**tracked in git**, ~39 MB) |
| `src/squat_study/` | `features.py`, `data.py`, `models.py`, `train.py`, `report.py` |
| `scripts/` | `build_manifest.py`, `build_dataset.py`, `run_cv.py`, `run_ablation.py`, `setup_env.*` |
| `notebooks/` | `01_explore_data.ipynb`, `02_results.ipynb` |
| `results/runs/<time>_<name>/` | one folder per run: `tables/`, `figures/`, per-model folders, `config_used.yaml` |

## 3. Run order

### Step 0 (morning): restore p94-p99
`p94`...`p99` (persons N and O) are in KEY_INFO but their folders are missing from `Temp_Dataset\preprocessed_data\`
(the batch log says they were processed; p95 shows an error in the log). You have them in another folder:

1. **Copy** (do not move) the six folders `p94 ... p99` into `C:\Current_Mini_Project_Working\Temp_Dataset\preprocessed_data\`.
2. Check each folder has the 6 files the models need: `points3d.trc, advancedangles.csv, reps.csv, rep_features.csv, meta.json, results.json`
   (p95 especially, because of the error in the log).
3. (Optional) all recordings were made with the same settings, so no further check is needed. If a folder is incomplete,
   `build_manifest.py` simply excludes it and says why in its report.
4. Rebuild everything and the report (about 5 minutes in total). **No code changes are needed** and the 9-model crash test
   does not have to be repeated: step 5 below (the baselines) is the check that the new data works.

```powershell
cd <repo>\squat_error_study
C:\squat_venv\Scripts\python.exe scripts\build_manifest.py --dry-run     # look at the report first
C:\squat_venv\Scripts\python.exe scripts\build_manifest.py
C:\squat_venv\Scripts\python.exe scripts\build_dataset.py
C:\squat_venv\Scripts\python.exe scripts\make_data_report.py
```
This adds up to 6 recordings (up to 5 in the 6-class setting). Commit the new `reps_T64.npz` so the lab gets it with `git pull`.

5. **Check the new data works** (1 minute): run the cheap baselines and open the figures.
```powershell
C:\squat_venv\Scripts\python.exe scripts\run_cv.py --models decision_tree logreg random_forest xgboost --name baselines
```
6. Open `docs\data_quality_report.html`: persons N and O should now show 7 recordings each. If p94-p99 show a problem
   (e.g. "neck missing"), that is a data property, not a code bug; decide as for the other recordings.

**Morning checklist, in order:** (0) copy p94-p99 into `Temp_Dataset\preprocessed_data\` -> rebuild (3 scripts above) ->
(1) install jupyter (`pip install -r requirements.txt ...`, section 1) -> (2) baselines (step 5) ->
(3) on the lab PC: `git pull`, `setup_env.ps1 -Gpu`, then `run_cv.py --name main --seeds 0 1 2` and `run_ablation.py` ->
(4) open the figures / `notebooks/02_results.ipynb`.

### The runs

```powershell
# (already done once) copy recordings + manifest, build the compact dataset
C:\squat_venv\Scripts\python.exe scripts\build_manifest.py
C:\squat_venv\Scripts\python.exe scripts\build_dataset.py

# quick check that everything works (minutes)
C:\squat_venv\Scripts\python.exe scripts\run_cv.py --models decision_tree xgboost cnn1d --epochs 3 --splits 3 --name check

# the cheap baselines (about a minute)
C:\squat_venv\Scripts\python.exe scripts\run_cv.py --models decision_tree logreg random_forest xgboost --name baselines

# ALL models, 5-fold person-wise CV   (use the lab GPU)
C:\squat_venv\Scripts\python.exe scripts\run_cv.py --name main --seeds 0 1 2

# ablation of the transformer             (use the lab GPU)
C:\squat_venv\Scripts\python.exe scripts\build_dataset.py --T 32      # only needed for the T32 / T128 variants
C:\squat_venv\Scripts\python.exe scripts\build_dataset.py --T 128
C:\squat_venv\Scripts\python.exe scripts\run_ablation.py
```

Then open `notebooks/02_results.ipynb`, or just open the PNGs in `results/runs/<run>/figures/`.
Add `--final transformer` to `run_cv.py` to also train one model on all persons and save it for the product.

### How long?

Measured on this PC's CPU (2 epochs, 3 folds, 873 reps): decision tree 0.4 s, XGBoost 6 s, 1D-CNN 11 s, BiGRU 19 s,
ST-GCN 55 s, **transformer 460 s (~77 s per epoch)**. The full run (60 epochs, 5 folds) is therefore roughly
**6 hours for the transformer on this CPU** and minutes on a GPU. Use the lab GPU for `main` and `ablation`.

## 4. Decisions built in (change them in `configs/default.yaml`)

* **Split by person** (18 persons, A-S without `E_PANTS_MALE`): `StratifiedGroupKFold`, a person is never in two folds.
  Two training persons are held out each fold to choose the best epoch, so the test fold is never used for selection.
* **6 classes**: `abnormal_tempo` is removed from the study (protocol not followed): `build_manifest.py` excludes those recordings.
* **Excluded recordings** (`manifest.csv`): `E_PANTS_MALE` (7), the KEY_INFO rep errors (p4, p50, p70, p71, p76, p105), and p93 (no `reps.csv`).
* **Reps per recording** differ (9 to 17): every valid rep is kept, each recording counts equally in the loss, and results
  are reported per rep and per recording (vote over its reps).
* **Resampling to T = 64 frames** makes reps equal length; the 4 tempo numbers keep the speed information.
* **Reps shorter than 1 s or longer than 12 s** are dropped.

## 5. Known data problems (put these in the paper's limitations)

1. **Zero-filled joints.** Pose2Sim wrote joints it could not triangulate as exactly `(0,0,0)`. `features.py` treats them as
   missing and interpolates. No lower-body joint is missing in any rep after this.
2. **Neck / Head / shoulders missing for whole recordings** (Neck: 13 recordings, 147 reps, mostly persons B, D, F). The pipeline's
   trunk angles (`sagittal_flexion_trunk`, `head_forward_angle`) are wrong there (values near 0 instead of up to ~28 deg).
   `data.max_trunk_missing: 0.5` drops those reps from training; set it to `null` to keep them. This matters most for `forward_lean`.
3. **p35 had no Hip joint**; it is rebuilt from the two hip joints.
4. **Forward direction** is taken from the feet and "up" from ankle-to-hip, so none of this needs the Neck.
5. **View ablation is simulated** (drops one body axis from the 3D joints), not a real single-camera experiment.
6. **p94-p99 missing from `Temp_Dataset`** (see Step 0). Until they are copied back, persons N and O have 3-4 recordings instead of 7.
   Per-recording quality for everything: `docs/data_quality_report.html` and `data/recording_quality.csv`.

## 6. Still to do

* Check angle accuracy against a reference (Kinovea on a few frames, Bland-Altman plot) and cite the bias.
* Run `main` and `ablation` on the lab GPU, look at the figures, choose the final model (best macro-F1, ties -> the one with explainable alpha).
* The saliency painting and the mesh viewer live in `two_view/` and `final_product/` and are not touched here.
