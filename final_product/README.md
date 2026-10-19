# Final product (front + side video &rarr; errors per rep, 3D body, synced videos)

## Quick start (Windows)

```powershell
git clone -b final_product https://github.com/DYNAMAXD/MarkerLess-Squat-Analysis-and-Real-Time-Posture-Correction-for-Injury-Prevention-using-Deep-Learning
```

then right-click `final_product\final_product.ps1` &rarr; **Run with PowerShell** (or
`powershell -ExecutionPolicy Bypass -File final_product\final_product.ps1`). Nothing else to install by hand:

* **First start** (needs internet, ~15-25 min, ~6 GB on C:): `setup.ps1` installs Python 3.11 if missing (winget),
  downloads the Git LFS files (MotionBERT weights, SMPL body model) if the clone only has pointers, and creates the three
  environments outside the repo (the repo's long folder name can hit the Windows path-length limit):
  `C:\Proj1Venv` (app + 3D body mesh, `requirements_app.txt`), `C:\Pose2SimVenv` (sync, calibration, pose, 3D;
  `two_view\requirements_pose2sim.txt`, onnxruntime-gpu when an NVIDIA GPU with driver &ge; 580 is present) and
  `C:\SquatModelVenv` (error model, `requirements_model.txt` + CPU torch). Existing environments are kept.
  The first analysis also downloads the pose models (~300 MB).
* **Later starts** only check the environments (a few seconds) and open http://127.0.0.1:5090. Keep the window open
  while an analysis runs.
* Already in git: the trained error model (`squat_model_export/`) and the dataset's camera calibration
  (`presets/cameras/Dataset_cameras`, offered as a saved calibration on the first start). Not in git: `outputs\`
  (your sessions, calibrations and Pose2Sim work folders are created there on this PC).

Options: `-PoseMode lightweight|balanced|performance` (default performance, the same as the dataset batch;
lightweight is about 3&times; faster on a CPU), `-Device auto|cpu|cuda`, `-Port`, `-ForceCpu` (setup: no GPU
packages), `-AppVenv` / `-Pose2SimVenv` / `-ModelVenv` (other environment folders).

## Using it

1. **New analysis**, one page in three steps. (1) **Camera calibration**, above the videos: a saved calibration,
   new intrinsic videos (+ optional extrinsic videos as a fallback) or an imported Pose2Sim `calibration.toml`;
   new ones are saved for the next analyses and calibrated first (the session waits; Stop stops both). The saved
   calibration's fx/fy/cx/cy are shown with sanity warnings. (2) **Squat videos**: front and side (board held
   still &rarr; clap &rarr; squats). Every video box takes drag &amp; drop, a click, or a path on this PC (typed or
   pasted with Ctrl+V; used in place, no upload). (3) **Start**. Jobs run one at a time in the background with a
   live stage list; Stop / Retry (finished steps are skipped).
2. **Results**: the MotionBERT body (drag = rotate, right/shift-drag = pan, wheel/pinch = zoom, double click =
   reset; Front/Side/Back/Top) coloured with the painted template of each rep's predicted error &times; the model's
   smoothed attention; the front and side keypoint videos (click to enlarge); errors and fixes (all reps / this
   rep); a timeline with the reps coloured by error and the attention strip (click a rep to jump; keys Space,
   &larr;/&rarr;, N/P, L); score, most frequent error and a "focus next time" line; **Compare reps** (two bodies
   in step, aligned start / bottom / end); downloads (`reps.csv` = rep, start, end in seconds; a zip with
   `rep_results.csv`, `predictions.json`, angles, 3D points, videos).
3. **History**: every session with % correct, errors per session, squat depth and rep duration over time.

**Error model**: the CNN-LSTM with attention pooling from `research/` (6 classes, macro-F1 0.76 in 5-fold
cross-validation by person; see `research/README.md`), shipped in git in `final_product\squat_model_export\`. To swap
in a newly trained one (`research\notebooks\04_final_model.ipynb`), copy `research\results\04_final\export\` there.
Its environment is `C:\SquatModelVenv` (created by `setup.ps1` with CPU torch; `research\setup_env.ps1` turns it into
the full research environment). The pipeline runs
`python -m squat_research.predict <session> --model <export>` (PYTHONPATH = `research\`, env `SQUAT_MODEL_CODE` /
`SQUAT_MODEL_DIR` / `SQUAT_MODEL_PY`) and uses its `predictions.json`: per rep the error, confidence, probabilities and
the frame attention α; the 3D body is coloured with the painted template of the error x α x confidence over time.
Without an installed model the sessions show the reps without errors; **Try demo predictions** (per session) fills in
made-up errors, clearly marked DEMO.

Saliency templates are read from `two_view\saliency_templates\` and `saliency_maps\` (env `SALIENCY_TEMPLATES`
overrides), matched to the error name. Texts, colours and the default camera view per error:
`product_content.json`. Data: `outputs\product\` (`cameras\`, `sessions\<id>\`, `_work\` = Pose2Sim projects).

## Files

| File | Purpose |
|---|---|
| `final_product.ps1` | Launcher (runs `setup.ps1`, starts the app, opens the browser) |
| `setup.ps1` | First-start setup / check: Git LFS files, Python 3.11, the three environments |
| `requirements_app.txt`, `requirements_model.txt` | Packages of `C:\Proj1Venv` and `C:\SquatModelVenv` (Pose2Sim: `two_view\requirements_pose2sim.txt`) |
| `presets/cameras/` | Camera calibrations copied into `outputs\product\cameras\` on the first start (the dataset's DJI pair) |
| `product_app.py` | Flask app: pages, camera calibrations (made with a session or reused), sessions, job queue (one job at a time), downloads |
| `product_pipeline.py` | Background jobs: session (preprocess &rarr; mesh &rarr; model &rarr; summary) and camera profile |
| `product_predictions.py` | `predictions.json` contract: inactive / demo predictions, attention smoothing, `rep_results.csv`, summary |
| `product_content.json` | Error names, colours, explanations, fixes and default camera view per error |
| `web/product.html`, `web/product.js`, `web/vendor/three.min.js` | The page (three.js r128; IndiGo-style indigo palette) |
| `web/vendor/fonts/` | Poppins (latin, 400-700, SIL OFL - `OFL.txt`), served locally so the page also works offline |
| `squat_model_export/` | the installed model export (`model.pt`, `config.json`, `norm_stats.json`, `model_card.md`; in git, `model.pt` as a normal file - not LFS) |
