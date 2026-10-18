# research: squat error classification

Four notebooks take the preprocessed two-camera recordings (output of `two_view/batch_preprocess.py`) to a chosen,
ablated and exported per-rep error model, which the final product (`final_product/`) uses. Every step is evaluated with
**5-fold cross-validation grouped by person, repeated with 3 seeds** (15 runs per model or variant).

## Results

6 classes (`correct`, `insufficient_depth`, `forward_lean`, `knee_valgus`, `knee_varus`, `heel_lift`), 18 persons,
103 recordings, 933 reps. Chance level about 0.17 macro-F1.

| Model | Macro-F1 per rep | Macro-F1 per recording | Params |
|---|---|---|---|
| **CNN-LSTM + attention (chosen)** | **0.758 ± 0.108** | **0.777** | 155 k |
| 1D-CNN + attention | 0.747 ± 0.089 | 0.763 | 207 k |
| TCN + attention | 0.734 ± 0.122 | 0.744 | 119 k |
| BiGRU + attention | 0.717 ± 0.100 | 0.754 | 169 k |
| BiLSTM + attention | 0.693 ± 0.078 | 0.737 | 210 k |
| ST-Transformer | 0.677 ± 0.093 | 0.689 | 290 k |
| XGBoost / random forest / logistic regression | 0.527 / 0.508 / 0.460 | | |

The CNN-LSTM is significantly better than the BiLSTM, the ST-Transformer and the classical baselines (paired Wilcoxon,
p < 0.01) and statistically tied with the 1D-CNN and TCN. Ablation (CNN-LSTM, full model 0.761): one camera only
-0.26 / -0.08, camera-world axes instead of the body frame -0.18, angles only -0.14, mean instead of attention pooling
-0.06, without the knee-offset / heel features -0.055, no rep trimming -0.03; with 33 / 50 / 75 % of the training
persons 0.62 / 0.67 / 0.69 (more participants would still help). No variant beat the defaults.

Tables, figures and one report per model are in `results/`; the executed notebooks show everything with outputs.

## Notebooks

| Notebook | What it does | Time (RTX 3050 laptop) |
|---|---|---|
| `notebooks/01_preprocessing_and_dataset.ipynb` | builds the dataset, 19 preprocessing correctness checks, data figures; no training | ~1 min |
| `notebooks/02_model_comparison.ipynb` | 3 classical baselines + 6 neural models with attention pooling; one report per model; α check; fixed selection rule; paired tests | ~65 min |
| `notebooks/03_ablation_best_model.ipynb` | ablation of the chosen model: inputs, cameras, preprocessing, architecture, pooling, training, data size; best configuration | ~60 min |
| `notebooks/04_final_model.ipynb` | final training on all persons for the median best epoch, export, inference with α on the real frames, unseen-person check | ~1 min |

## Setup and run

1. `powershell -ExecutionPolicy Bypass -File research\setup_env.ps1` creates `C:\SquatModelVenv` (torch with CUDA when an
   NVIDIA GPU is present, the packages of `requirements.txt`) and the Jupyter kernel **Python (SquatModelVenv)**.
2. Set the data paths in `config/default.yaml` (`paths.recordings`: the batch's `preprocessed_data/`, `paths.key_info`:
   `KEY_INFO.xlsx` with the person of every recording).
3. Run the notebooks in order (Run All), or unattended:
   `C:\SquatModelVenv\Scripts\python.exe research\scripts\run_notebooks.py 01 02 03 04`.
   Training runs in its own process; the notebooks show live progress (folds done, epoch, ETA) and can be interrupted
   without stopping it.
4. Install the model in the product: copy `results/04_final/export/` to `final_product/squat_model_export/`.

Not in git (re-created by running the notebooks): `data/` (dataset files), the per-fold runs with checkpoints
(`results/02_model_comparison/cv/`, `results/03_ablation/runs/`) and model weights (`*.pt`, `*.joblib`).

## Method

* **Data**: abnormal_tempo dropped (people did not follow the speed protocol); person E_PANTS_MALE excluded (poor
  tracking; used as an unseen-person check), p01 and the KEY_INFO error recordings excluded; reps with a lower-body joint
  missing > 25 % or the Neck missing > 50 % of their frames dropped.
* **Preprocessing** (`squat_research/features.py`): Pose2Sim's (0,0,0) joints = missing, interpolated; body frame from
  the real standing direction (ankles -> hips) and the feet, hip-centred, divided by leg length; left / right check;
  reps trimmed to the movement (standing level taken per side of the bottom), resampled to 64 frames. Inputs: joint
  tokens 64 x 22 x 6 (position + velocity), frame features 64 x 19 (14 angles + knee offset, heel height, hip height),
  tempo 4.
* **Evaluation**: StratifiedGroupKFold by person (k = 5) x 3 seeds; 2 validation persons per fold for early stopping
  (patience 20, best-epoch checkpoints); normalisation from training data only; classes balanced and every recording
  weighted equally; macro-F1 per rep (main), per recording, per class, confusion matrices, paired Wilcoxon tests.
* **Model choice**: neural models whose α passes the deletion test (masking the top-α frames lowers the true-class
  probability at least 2x more than random frames) -> best mean macro-F1 -> ties within one std broken by
  recording-level macro-F1, then size.

## Code

```
research/
  setup_env.ps1, requirements.txt
  config/default.yaml          every setting
  squat_research/              manifest, features, checks, dataset, models, engine (CV, alpha check, selection,
                               final fit, export, predict), report, monitor, viz, predict (CLI for the product)
  scripts/                     run_models.py, run_ablation.py (background runners), run_notebooks.py, make_notebooks*.py
  notebooks/                   01-04
  results/                     tables, figures, model reports, best configuration, final model card
```
