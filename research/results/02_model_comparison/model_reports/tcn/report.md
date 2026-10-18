## TCN  (convolutional)

**Input.** per frame: 151 channels, 64 frames.

**Processing.** input projection to 64 channels, 4 residual blocks of two dilated Conv1d (kernel 3, dilation 1, 2, 4, 8; non-causal): the receptive field covers the whole 64-frame rep; attention pooling -> alpha; + tempo -> classifier.

**Size.** 118,598 trainable parameters. Training: AdamW (lr 0.001, weight decay 0.01), cosine schedule, up to 60 epochs, early stopping on the validation persons (patience 20), label smoothing 0.05, noise augmentation 0.01.

**Common preprocessing.** 3D joints from Pose2Sim -> body frame (lateral / up / forward from the standing posture and the feet), hip-centred, divided by the leg length; reps trimmed to the movement and resampled to 64 frames; z-scored with training-fold statistics.

**Performance (5-fold cross-validation by person x 3 seeds = 15 runs).**

| metric | value |
|---|---|
| macro-F1 per rep | 0.734 ± 0.122 |
| accuracy per rep | 0.750 ± 0.112 |
| macro-F1 per recording (vote) | 0.744 ± 0.145 |
| best / worst run | 0.923 / 0.480 |
| alpha check: drop when the top-10 % alpha frames are masked vs random | 0.050 vs 0.008 |
| median best epoch | 10 |
| training time (15 runs) | 1.5 min |

**F1 per class.**

| class              |    F1 |   reps |
|:-------------------|------:|-------:|
| correct            | 0.578 |    402 |
| insufficient depth | 0.634 |    399 |
| forward lean       | 0.767 |    510 |
| knee valgus        | 0.873 |    432 |
| knee varus         | 0.804 |    489 |
| heel lift          | 0.827 |    567 |

Weakest class: **correct** (F1 0.58). Most frequent confusion: **correct -> insufficient depth** (85 reps).

![confusion](confusion.png)
![learning curves](learning_curves_tcn.png)
![alpha profile](alpha_profile_tcn.png)