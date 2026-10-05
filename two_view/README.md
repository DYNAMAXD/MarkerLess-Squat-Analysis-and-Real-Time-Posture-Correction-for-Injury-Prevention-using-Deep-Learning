# two_view — front + side camera preprocessing

Runs the existing single-view pipeline on a front video and a side video, aligns them
with the clap sound, and fuses the two angle tables **per squat repetition**.

```
front.mp4 ─┐  clap sync (sync_clap.py): offset = t_clap_side - t_clap_front
           ├─► existing pipeline on each video ─► extracted_data.csv (front / side)
side.mp4  ─┘                                              │
                                                          ▼
                                  rep_fusion.py: detect reps on knee flexion, match rep N <-> rep N
                                                          ▼
                                  rep_features.csv  (1 row per rep: front_* and side_* features)
                                  report.html       (both views on one time axis + rep table)
```

No video editing is needed; the clap is only used to measure the time offset.
No camera calibration is needed; views are fused at the feature level, not in 3D.

## Website (easiest)

Run `two_view\start_website.ps1` (right-click > Run with PowerShell). First run installs the environment at
`C:\Proj1Venv` (~10 min); later runs just open http://127.0.0.1:5000. Upload a front and a side video: the page
syncs them with the clap, runs the pipeline, then shows both videos together, angle graphs (click a curve to jump
to that moment), a per-rep table and downloads. Past analyses are listed on the left.
Files: `web_app.py` (server), `web/index.html` (page), `web_results.py` (packages results for the page).

## Run from the command line

```
pip install imageio-ffmpeg
python two_view/run_two_view.py --front front.mp4 --side side.mp4 --name subj01_set1 [--device cpu|gpu|auto]
```

Outputs go to `outputs/two_view/<name>/`: `rep_features.csv`, `report.html`, `sync.json`.
The per-view pipeline outputs are in `outputs/<name>_front/` and `outputs/<name>_side/`.

Already have the two `extracted_data.csv` files?

```
python two_view/run_two_view.py --front-csv A.csv --side-csv B.csv --fps-front 29.97 --fps-side 29.97 --offset -0.39 --name test
```

## Files

| File | Purpose |
|---|---|
| `sync_clap.py` | Finds the clap in each video's audio, returns the time offset and a confidence score |
| `rep_fusion.py` | Detects reps, builds per-rep features for each view, matches reps across views |
| `two_view_report.py` | Writes the interactive HTML report |
| `run_two_view.py` | Runs everything end to end |

## Limitations

- Needs a clear clap in both videos (a confidence below 8 triggers a warning).
- Reps are found from knee flexion; slow or partial reps below 20° prominence are ignored.
- Fusion is at feature level; there is no shared 3D coordinate system between views.
