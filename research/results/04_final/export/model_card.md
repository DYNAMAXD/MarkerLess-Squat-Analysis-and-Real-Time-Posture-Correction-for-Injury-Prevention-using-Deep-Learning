# Squat error model: CNN-LSTM

* Classes: correct, insufficient_depth, forward_lean, knee_valgus, knee_varus, heel_lift
* Data: 933 reps, 91 recordings, 18 persons (two-camera Pose2Sim, 29.97 fps)
* Configuration: `full` (defaults); trained on all persons for 15 epochs
* Cross-validation (k = 5 by person, 3 seeds): macro-F1 0.761 ± 0.106, recording level 0.790
* Parameters: 155,398
* Output per rep: class probabilities and alpha (64 frame-attention weights) -> alpha on the real frames for the 3D body
* Files: model.pt (weights), config.json (model, data and feature settings), norm_stats.json (z-score statistics)
* Limitations: 18 persons; the person is the main source of error; abnormal_tempo is not modelled.
