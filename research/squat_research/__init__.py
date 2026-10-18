"""
squat_research - the code behind the research notebooks (research_folders/notebooks).

    common.py    config loading, paths, small helpers
    manifest.py  which recordings are used (KEY_INFO persons, labels, exclusions)
    features.py  one recording -> per-rep arrays (cleaning, body frame, trimming, resampling, features)
    dataset.py   build / load the compact dataset, person-wise folds, normalisation, ablation switches
    models.py    4 base models + 7 attention models (every attention model returns alpha)
    engine.py    cross-validation runner with progress file, alpha deletion test, final fit, export, predict
    monitor.py   live view of a running experiment (for the notebooks)
    viz.py       every figure, saved as PNG into results/
"""
FEATURE_VERSION = 4   # 4: standing level per side of the bottom in the rep trimming
