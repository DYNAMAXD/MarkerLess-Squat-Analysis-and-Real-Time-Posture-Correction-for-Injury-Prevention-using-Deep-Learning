"""
dataset.py - the compact dataset every model trains from, person-wise folds, normalisation, ablation switches.

    build_dataset(cfg)   manifest's included recordings -> data/processed/reps_T<T>[_notrim].npz (+ .json)
    load_dataset(cfg)    -> dict of arrays restricted to the classes, quality filters applied, y = class index
    make_folds(...)      StratifiedGroupKFold by PERSON + validation persons inside the training part
    Normaliser           z-score statistics from the training part only
    apply_switches(...)  zero input parts for the ablation study
"""
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import FEATURE_VERSION
from .common import path
from .features import FRAME_FEATURE_NAMES, JOINT_CHANNELS, TEMPO_NAMES, recording_to_reps
from .manifest import build_manifest


def dataset_file(cfg):
    d = cfg["data"]
    suffix = ("" if d.get("trim", True) else "_notrim") + ("" if d.get("body_frame", True) else "_worldaxes") + \
             ("" if d.get("leg_norm", True) else "_metres")
    return path(cfg, "processed") / f"reps_T{d['T']}{suffix}.npz"


def build_dataset(cfg, log=print):
    man = build_manifest(cfg)
    path(cfg, "manifest").parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(path(cfg, "manifest"), index=False)
    inc = man[man["include"]]
    rec_dir = path(cfg, "recordings")
    cols = {k: [] for k in ("joints", "frame", "tempo", "summary", "flex", "label", "person", "recording", "rep_no",
                            "start", "end", "bottom", "duration_s", "missing", "trunk_missing")}
    infos, summary_names, t0 = {}, None, time.time()
    for _, r in inc.iterrows():
        reps, info = recording_to_reps(rec_dir / r["id"], cfg["data"])
        if summary_names is None and info["summary_names"]:
            summary_names = info["summary_names"]
        pos = {n: i for i, n in enumerate(info["summary_names"])}
        infos[r["id"]] = {k: v for k, v in info.items() if k not in ("joint_names", "summary_names")}
        joint_names = info["joint_names"]
        for it in reps:
            s = it["summary"]
            row = np.array([s[pos[n]] if s is not None and n in pos else np.nan for n in summary_names], np.float32)
            for k in ("joints", "frame", "tempo", "flex", "rep_no", "start", "end", "bottom", "duration_s", "missing", "trunk_missing"):
                cols[k].append(it[k])
            cols["summary"].append(row)
            cols["label"].append(r["label"])
            cols["person"].append(r["person"])
            cols["recording"].append(r["id"])
    arrays = {k: np.stack(v) if k in ("joints", "frame", "tempo", "summary", "flex") else np.array(v) for k, v in cols.items()}
    out = dataset_file(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = dict(feature_version=FEATURE_VERSION, data=cfg["data"], joint_names=joint_names, joint_channels=JOINT_CHANNELS,
                frame_feature_names=FRAME_FEATURE_NAMES, tempo_names=TEMPO_NAMES, summary_names=summary_names,
                n_reps=int(len(arrays["label"])), recordings=infos, built=time.strftime("%Y-%m-%d %H:%M:%S"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    log(f"{len(inc)} recordings -> {len(arrays['label'])} reps in {time.time() - t0:.0f} s -> {out}")
    return man, out


def load_dataset(cfg, filters=True):
    f = dataset_file(cfg)
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run build_dataset(cfg) first (notebook 01)")
    d = np.load(f, allow_pickle=False)
    meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(f"{f.name} was built with feature version {meta.get('feature_version')}; rebuild it")
    classes = cfg["manifest"]["classes"]
    keep = np.isin(d["label"], classes)
    if filters:
        keep &= d["missing"] <= cfg["data"]["max_missing_core"]
        keep &= d["trunk_missing"] <= cfg["data"]["max_trunk_missing"]
    out = {k: d[k][keep] for k in d.files}
    out["y"] = np.array([classes.index(c) for c in out["label"]], dtype=np.int64)
    out["classes"] = list(classes)
    out["meta"] = meta
    out["n_filtered"] = int((~keep & np.isin(d["label"], classes)).sum())
    return out


UPPER_BODY = ["Neck", "Head", "Nose", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]


def apply_switches(joints, frame, tempo, inputs="both", velocity=True, tempo_on=True, view="both", joint_mask=None, frame_mask=None):
    """Zero the input parts an ablation variant should not see (after z-scoring, zero = 'no information').
    joint_mask / frame_mask: boolean arrays (True = zero this joint / frame feature)."""
    joints, frame, tempo = joints.copy(), frame.copy(), tempo.copy()
    if joint_mask is not None and np.any(joint_mask):
        joints[:, :, np.asarray(joint_mask)] = 0
    if frame_mask is not None and np.any(frame_mask):
        frame[:, :, np.asarray(frame_mask)] = 0
    if inputs == "frame":
        joints[...] = 0
    elif inputs == "joints":
        frame[...] = 0
    if not velocity:
        joints[..., 3:] = 0
    if not tempo_on:
        tempo[...] = 0
    if view != "both":                     # simulated single camera: drop one body axis (side drops lateral)
        drop = 0 if view == "side" else 2
        joints[..., drop] = 0
        joints[..., 3 + drop] = 0
        frame[...] = 0
    return joints, frame, tempo


def switches(cfg, joint_names=None):
    """Keyword arguments for apply_switches from the config's data section."""
    d = cfg["data"]
    sw = dict(inputs=d["inputs"], velocity=d["velocity"], tempo_on=d["tempo"], view=d["view"])
    if joint_names is not None and d.get("joints", "all") == "lower_body":
        sw["joint_mask"] = np.array([n in UPPER_BODY for n in joint_names])
    drop = d.get("drop_frame_features") or []
    if drop:
        from .features import FRAME_FEATURE_NAMES
        sw["frame_mask"] = np.array([n in drop for n in FRAME_FEATURE_NAMES])
    return sw


def make_folds(y, person, n_splits=5, seed=0, val_persons=2):
    """Yields (train, val, test) rep indices; a person is never in two of them."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups=person):
        tr_p = np.unique(person[tr])
        n_val = min(val_persons, max(len(tr_p) - 1, 0))
        val_p = rng.choice(tr_p, size=n_val, replace=False) if n_val else []
        is_val = np.isin(person[tr], val_p)
        yield tr[~is_val], tr[is_val], te


class Normaliser:
    """Per-channel z-score, fitted on training reps only."""

    def fit(self, joints, frame, tempo):
        self.j_mu, self.j_sd = joints.mean(axis=(0, 1, 2)), joints.std(axis=(0, 1, 2)) + 1e-6
        self.f_mu, self.f_sd = frame.mean(axis=(0, 1)), frame.std(axis=(0, 1)) + 1e-6
        self.t_mu, self.t_sd = tempo.mean(axis=0), tempo.std(axis=0) + 1e-6
        return self

    def __call__(self, joints, frame, tempo):
        return (((joints - self.j_mu) / self.j_sd).astype(np.float32), ((frame - self.f_mu) / self.f_sd).astype(np.float32),
                ((tempo - self.t_mu) / self.t_sd).astype(np.float32))

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("j_mu", "j_sd", "f_mu", "f_sd", "t_mu", "t_sd")}

    @classmethod
    def from_dict(cls, d):
        n = cls()
        for k, v in d.items():
            setattr(n, k, np.asarray(v, dtype=np.float32))
        return n
