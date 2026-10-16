# Final product (front + side video &rarr; errors per rep, 3D body, synced videos)

**Run** `final_product\final_product.ps1` &rarr; http://127.0.0.1:5090 (uses `C:\Proj1Venv` for the app and the mesh,
`C:\Pose2SimVenv` for the preprocessing; keep the window open while an analysis runs). Options:
`-PoseMode lightweight|balanced|performance` (default performance, the same as the dataset batch; lightweight is
about 3&times; faster on a CPU), `-Device auto|cpu|cuda`, `-Port`.

1. **Cameras**: make a camera profile once, from the two intrinsic videos (+ optional extrinsic videos as a fallback
   calibration) or by importing a Pose2Sim `calibration.toml`. It shows fx/fy/cx/cy with sanity warnings.
2. **New analysis**: pick the profile, give the front and side squat videos (board held still &rarr; clap &rarr;
   squats; upload or type a path). Jobs run one at a time in the background with a live stage list; Stop / Retry
   (finished steps are skipped).
3. **Results**: the MotionBERT body (drag = rotate, right/shift-drag = pan, wheel/pinch = zoom, double click =
   reset; Front/Side/Back/Top) coloured with the painted template of each rep's predicted error &times; the model's
   smoothed attention; the front and side keypoint videos (click to enlarge); errors and fixes (all reps / this
   rep); a timeline with the reps coloured by error and the attention strip (click a rep to jump; keys Space,
   &larr;/&rarr;, N/P, L); score, most frequent error and a "focus next time" line; **Compare reps** (two bodies
   in step, aligned start / bottom / end); downloads (`reps.csv` = rep, start, end in seconds; a zip with
   `rep_results.csv`, `predictions.json`, angles, 3D points, videos).
4. **History**: every session with % correct, errors per session, squat depth and rep duration over time.

**The error model is not trained yet**: sessions show the reps without errors. **Try demo predictions** (per
session) fills in made-up errors and attention, clearly marked DEMO, to try the page. Once the model exists
(`final_product\squat_model_export\model.pt` + `C:\SquatModelVenv`, see `ai_lead_handoff/`), the pipeline
runs `python -m squat_model.predict <session> --model <export>` (PYTHONPATH = `two_view\`, env `SQUAT_MODEL_CODE`) and uses its `predictions.json` (contract in
`product_predictions.py`).

Saliency templates are read from `two_view\saliency_templates\` and `saliency_maps\` (env `SALIENCY_TEMPLATES`
overrides), matched to the error name. Texts, colours and the default camera view per error:
`product_content.json`. Data: `outputs\product\` (`cameras\`, `sessions\<id>\`, `_work\` = Pose2Sim projects).

## Files

| File | Purpose |
|---|---|
| `final_product.ps1` | Launcher (checks the environments, starts the app, opens the browser) |
| `product_app.py` | Flask app: pages, camera profiles, sessions, job queue (one job at a time), downloads |
| `product_pipeline.py` | Background jobs: session (preprocess &rarr; mesh &rarr; model &rarr; summary) and camera profile |
| `product_predictions.py` | `predictions.json` contract: inactive / demo predictions, attention smoothing, `rep_results.csv`, summary |
| `product_content.json` | Error names, colours, explanations, fixes and default camera view per error |
| `web/product.html`, `web/product.js`, `web/vendor/three.min.js` | The page (three.js r128) |
| `squat_model_export/` | (not yet) the trained model's export folder |
