"""
data.py   (MODULE)
==================
Loading the compact .npz, applying the input switches used by the ablation study,
person-wise splits and per-fold normalisation.

Sections
--------
 1. load_dataset        read reps_T<T>.npz, keep the chosen classes
 2. apply_switches      inputs / velocity / tempo / view switches (ablation)
 3. make_folds          StratifiedGroupKFold by PERSON + a validation split inside the training part
 4. Normaliser          z-score statistics from the TRAINING part only
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_dataset(npz_path, classes, data_cfg=None):
    """
    Returns a dict of numpy arrays restricted to `classes`, with integer labels y.
    data_cfg (the "data" section of the config) may set
        max_missing_core   drop reps whose lower-body joints are missing in more frames than this
        max_trunk_missing  drop reps whose Neck is missing in more frames than this
                           (the pipeline trunk angles are unreliable there)
    """
    d = np.load(npz_path, allow_pickle=False)
    meta = json.load(open(Path(npz_path).with_suffix(".json")))
    keep = np.isin(d["label"], classes)
    cfg = data_cfg or {}
    if cfg.get("max_missing_core") is not None and "missing" in d.files:
        keep &= d["missing"] <= cfg["max_missing_core"]
    if cfg.get("max_trunk_missing") is not None and "trunk_missing" in d.files:
        keep &= d["trunk_missing"] <= cfg["max_trunk_missing"]
    out = {k: d[k][keep] for k in d.files}
    cls_to_i = {c: i for i, c in enumerate(classes)}
    out["y"] = np.array([cls_to_i[c] for c in out["label"]], dtype=np.int64)
    out["classes"] = list(classes)
    out["meta"] = meta
    return out


# ----------------------------------------------------------------------------
# 2. Input switches (used by the ablation study)
# ----------------------------------------------------------------------------
def apply_switches(joints, frame, tempo, inputs="both", velocity=True, tempo_on=True, view="both"):
    """
    Zeroes the parts of the input that an ablation variant should not see.
    (Zeroing AFTER z-scoring means "the feature carries no information".)
    joints [N,T,J,6]  channels: pos lat/up/fwd, vel lat/up/fwd
    frame  [N,T,F]
    tempo  [N,4]
    """
    joints, frame, tempo = joints.copy(), frame.copy(), tempo.copy()
    if inputs == "frame":
        joints[...] = 0
    elif inputs == "joints":
        frame[...] = 0
    if not velocity:
        joints[..., 3:] = 0
    if not tempo_on:
        tempo[...] = 0
    if view != "both":
        # simulated single-camera view = orthographic projection: drop one body axis
        # side view sees up + forward (drops lateral); front view sees lateral + up (drops forward)
        drop = 0 if view == "side" else 2
        joints[..., drop] = 0
        joints[..., 3 + drop] = 0
        frame[...] = 0                      # angle features mix axes, so they are removed in view ablations
    return joints, frame, tempo


# ----------------------------------------------------------------------------
# 3. Folds
# ----------------------------------------------------------------------------
def make_folds(y, person, n_splits=5, seed=0, val_persons=2):
    """
    Yields (train_idx, val_idx, test_idx) over reps.
    * test  = one StratifiedGroupKFold fold (a person is NEVER in two folds)
    * val   = `val_persons` persons taken out of the remaining training part, used only to choose
              the best epoch / stop early, so the test fold is never used for model selection
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups=person):
        tr_persons = np.unique(person[tr])
        n_val = min(val_persons, max(len(tr_persons) - 1, 0))
        val_p = rng.choice(tr_persons, size=n_val, replace=False) if n_val else []
        is_val = np.isin(person[tr], val_p)
        yield tr[~is_val], tr[is_val], te


# ----------------------------------------------------------------------------
# 4. Normalisation (statistics from the training part only)
# ----------------------------------------------------------------------------
class Normaliser:
    """Per-channel z-score. Fit on training reps, apply to anything."""

    def fit(self, joints, frame, tempo):
        self.j_mu = joints.mean(axis=(0, 1, 2)); self.j_sd = joints.std(axis=(0, 1, 2)) + 1e-6
        self.f_mu = frame.mean(axis=(0, 1));     self.f_sd = frame.std(axis=(0, 1)) + 1e-6
        self.t_mu = tempo.mean(axis=0);          self.t_sd = tempo.std(axis=0) + 1e-6
        return self

    def __call__(self, joints, frame, tempo):
        return ((joints - self.j_mu) / self.j_sd).astype(np.float32), \
               ((frame - self.f_mu) / self.f_sd).astype(np.float32), \
               ((tempo - self.t_mu) / self.t_sd).astype(np.float32)

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in
                ("j_mu", "j_sd", "f_mu", "f_sd", "t_mu", "t_sd")}
