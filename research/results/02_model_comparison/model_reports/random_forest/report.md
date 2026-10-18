## Random forest  (classical baseline)

**Input.** summary row (46) + tempo (4) = 50 values per rep.

**Processing.** median imputation, 300 decision trees (bootstrap), majority of tree probabilities.

**Preprocessing.** The summary row is computed by the Pose2Sim preprocessing pipeline per rep (`rep_features.csv`, statistics of its joint angles); the 4 tempo numbers come from the trimmed rep. No body-frame joints are used.

**Performance (5-fold cross-validation by person x 3 seeds = 15 runs).**

| metric | value |
|---|---|
| macro-F1 per rep | 0.508 ± 0.052 |
| accuracy per rep | 0.529 ± 0.049 |
| macro-F1 per recording (vote) | 0.517 ± 0.067 |
| best / worst run | 0.608 / 0.427 |
| training time (15 runs) | 0.2 min |

**F1 per class.**

| class              |    F1 |   reps |
|:-------------------|------:|-------:|
| correct            | 0.256 |    402 |
| insufficient depth | 0.593 |    399 |
| forward lean       | 0.601 |    510 |
| knee valgus        | 0.501 |    432 |
| knee varus         | 0.502 |    489 |
| heel lift          | 0.647 |    567 |

Weakest class: **correct** (F1 0.26). Most frequent confusion: **knee varus -> correct** (127 reps).

![confusion](confusion.png)