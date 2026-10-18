# Final product (front + side video &rarr; errors per rep, 3D body, synced videos)

**Run** `final_product\final_product.ps1` &rarr; http://127.0.0.1:5090 (uses `C:\Proj1Venv` for the app and the mesh,
`C:\Pose2SimVenv` for the preprocessing; keep the window open while an analysis runs). Options:
`-PoseMode lightweight|balanced|performance` (default performance, the same as the dataset batch; lightweight is
about 3&times; faster on a CPU), `-Device auto|cpu|cuda`, `-Port`.

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
cross-validation by person; see `research/README.md`). Install it by copying `research\results\04_final\export\`
(train it with `research\notebooks\04_final_model.ipynb`) to `final_product\squat_model_export\`; its environment is
`C:\SquatModelVenv` (`research\setup_env.ps1`). The pipeline runs
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
| `final_product.ps1` | Launcher (checks the environments, starts the app, opens the browser) |
| `product_app.py` | Flask app: pages, camera calibrations (made with a session or reused), sessions, job queue (one job at a time), downloads |
| `product_pipeline.py` | Background jobs: session (preprocess &rarr; mesh &rarr; model &rarr; summary) and camera profile |
| `product_predictions.py` | `predictions.json` contract: inactive / demo predictions, attention smoothing, `rep_results.csv`, summary |
| `product_content.json` | Error names, colours, explanations, fixes and default camera view per error |
| `web/product.html`, `web/product.js`, `web/vendor/three.min.js` | The page (three.js r128; IndiGo-style indigo palette) |
| `web/vendor/fonts/` | Poppins (latin, 400-700, SIL OFL - `OFL.txt`), served locally so the page also works offline |
| `squat_model_export/` | the installed model export (`model.pt`, `config.json`, `norm_stats.json`, `model_card.md`; not in git) |
