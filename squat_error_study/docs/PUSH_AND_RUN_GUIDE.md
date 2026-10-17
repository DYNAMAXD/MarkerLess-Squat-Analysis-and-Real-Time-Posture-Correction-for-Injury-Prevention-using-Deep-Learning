# Push, run on the GPU / friend's laptop, compare, write the report (checklist)

Written 2026-10-17. Companion to `WORKFLOW_FOR_COMPARISON.md` (what the code does). This file is the order of work.
Branch name used below: `squat-error-study`.

## A. On this PC, before pushing

- [x] Raw data re-copied, p94-p99 and p93 restored; `abnormal_tempo` excluded; dataset rebuilt (103 recordings, 1,068 reps, 933 after filters).
- [x] Cheap baselines run (`results/runs/20261017-145728_baselines6`).
- [x] Ablation script fixed: `mean_pooling` is transformer-only; training noise no longer touches blanked channels.
- [ ] **Delete test leftovers** (they are not results). Run in PowerShell from `squat_error_study`:
  ```powershell
  Remove-Item -Recurse -Force results\runs\*_smoketest
  Remove-Item -Recurse -Force results\runs\20261017-144259_baselines      # made before the rebuild
  Remove-Item -Recurse -Force results\runs\20261017-150359_ablation_cnn1d  # 2-epoch plumbing test
  Remove-Item -Recurse -Force results\runs\20261017-150753_ablation_bigru_attn
  ```
  Keep `20261017-145728_baselines6`.
- [ ] Look at `docs\data_quality_report.html` once and decide whether any recording looks wrong to you.
- [ ] (Optional, ask first) decide on the rep-dropping idea; the current filters are already the "objective" ones.

## B. Create the branch and push (repo root)

```powershell
cd C:\Current_Mini_Project_Working\MarkerLess-Squat-Analysis-and-Real-Time-Posture-Correction-for-Injury-Prevention-using-Deep-Learning
git status                              # should show only squat_error_study/ as untracked
git checkout -b squat-error-study       # new branch from main
git add squat_error_study
git status                              # check: no .pt/.joblib, no data/recordings, about 40 MB
git commit -m "Add squat error classification study (6 classes, person-wise CV, ablation)"
git push -u origin squat-error-study
```
Then on GitHub open a Pull Request only if you want a review; the branch alone is enough for your friend to `git checkout` it.
Not pushed on purpose: `data/recordings/` (170 MB, copy by hand if someone needs to rebuild), weights, the venv.

## C. Run on the GPU machine (lab)

```powershell
git clone <repo-url>
cd <repo>
git checkout squat-error-study
cd squat_error_study
.\scripts\setup_env.ps1 -Gpu                      # prints the python path -> <PY>
<PY> -c "import torch; print(torch.cuda.is_available())"     # must print True
<PY> scripts\run_cv.py --models decision_tree xgboost cnn1d --epochs 3 --splits 3 --name check   # 2-3 min sanity check
<PY> scripts\run_cv.py --name main --seeds 0 1 2             # all 9 models, the main table
<PY> scripts\run_ablation.py --seeds 0 1 2                    # transformer ablation (T32/T128 are skipped unless built)
```
T32 / T128 need `data/recordings/` (copy it from this PC to the lab PC, then `<PY> scripts\build_manifest.py --src <Temp_Dataset path>` is NOT needed; only `build_dataset.py --T 32` and `--T 128` are). If you skip the copy, report ablation without T.
Add `--final transformer` to the `main` run if you also want a model trained on everyone for the product.

Save the outputs: the run folders are `results/runs/<time>_main` and `..._ablation_transformer`. Tables, figures and json are tracked;
commit them on the same branch and push (`git add results/runs; git commit -m "Lab GPU results"; git push`).
Weights (`*.pt`) are ignored: copy the best model folder back by hand (USB / drive).

## D. Your friend's laptop

1. `git clone` and `git checkout squat-error-study`, run `setup_env.ps1` (CPU is fine for the baselines and for a few epochs).
2. Open Claude Code in the repo and paste the prompt from section 0 of `WORKFLOW_FOR_COMPARISON.md`.
3. Work through the comparison checklist in section 8 of that file **before** comparing numbers (split, classes, recordings, input, model selection, metric, variance).
4. His model goes in as a new class in `models.py` with the same `(joints, frame, tempo) -> (logits, alpha)` signature, then
   `run_cv.py --models <his_model> --name friend` gives a run folder that is directly comparable.
5. He pushes his results to his own branch (`friend-results`), so nothing of yours is overwritten.

## E. After the runs: what goes into the report

1. Open `results/runs/<main>/figures` (confusion matrices, per-class F1, model comparison) and `tables`.
2. Pick the final model by recording-level and rep-level macro-F1 (mean +- std over 3 seeds); ties go to the model with explainable attention.
3. Ablation table: `full` vs each variant; only trust differences larger than about 0.05; compare `view_*` with `joints_only`.
4. Write the limitations from README section 5 + `WORKFLOW_FOR_COMPARISON.md` section 2 (Neck missing, excluded recordings, simulated views, small number of persons, p126/p127/p131/p132 sync uncertain).
5. Still to do: angle accuracy against Kinovea (Bland-Altman) and a literature search before claiming novelty.
6. Report outline: Introduction, Related work, Data (cameras, Pose2Sim, 18 persons, 6 classes), Method (features, models), Experiments (person-wise CV, baselines, ablation), Results, Explainability (alpha), Limitations, Conclusion.

## F. Things that can go wrong

| Problem | Cause / fix |
|---|---|
| `torch.cuda.is_available()` is False | CPU torch installed; rerun `setup_env.ps1 -Gpu` (Windows) or `bash scripts/setup_env.sh gpu` |
| pip fails on a long path | create the venv at a short path such as `C:\squat_venv` |
| `reps_T32.npz missing` | run `build_dataset.py --T 32` (needs `data/recordings/`) |
| `git push` rejected, file too large | something big got added: `git rm --cached <file>`, check `.gitignore`, amend the commit |
| numbers differ between machines | GPU vs CPU and library versions change results slightly; always compare mean +- std over 3 seeds, not single runs |
