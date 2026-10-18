"""
manifest.py - decides which recordings are used, and why the others are not.

One row per recording folder: id, person (KEY_INFO), label (meta.json, cleaned), include, reason.
Rules (config `manifest`): class list, excluded persons / ids, KEY_INFO file errors, required files present.
"""
import json
import re
from pathlib import Path

import pandas as pd

REQUIRED = ["points3d.trc", "advancedangles.csv", "reps.csv", "rep_features.csv", "meta.json", "results.json"]
LABEL_FIX = {"heels_up": "heel_lift", "heel_up": "heel_lift", "heel_raise": "heel_lift", "shallow": "insufficient_depth",
             "depth": "insufficient_depth", "valgus": "knee_valgus", "varus": "knee_varus", "lean": "forward_lean",
             "tempo": "abnormal_tempo", "good": "correct"}


def clean_label(raw):
    """'"correct"' -> 'correct', 'heels up' -> 'heel_lift' (stray quote characters come from CSV quoting)."""
    s = re.sub(r"[\s\-]+", "_", str(raw).strip().strip("\"'").strip().lower()).strip("_")
    return LABEL_FIX.get(s, s)


def read_key_info(path):
    """KEY_INFO.xlsx -> DataFrame(id, person, file_error)."""
    df = pd.read_excel(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return pd.DataFrame({"id": df["ID"].astype(str).str.strip(), "person": df["PERSON"].astype(str).str.strip(),
                         "file_error": df.get("FILE_ERROR(S)", pd.Series(["-"] * len(df))).fillna("-").astype(str).str.strip()})


def build_manifest(cfg):
    from .common import path
    rec_dir, key = path(cfg, "recordings"), read_key_info(path(cfg, "key_info"))
    persons = dict(zip(key["id"], key["person"]))
    errors = dict(zip(key["id"], key["file_error"]))
    m = cfg["manifest"]
    rows = []
    for d in sorted(p for p in rec_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rid = d.name
        meta = {}
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        raw = (meta.get("errors") or [meta.get("error_performed") or meta.get("label") or ""])
        labels = sorted({clean_label(x) for x in (raw if isinstance(raw, list) else [raw]) if str(x).strip()})
        label = labels[0] if len(labels) == 1 else None
        missing = [f for f in REQUIRED if not (d / f).exists()]
        reason = None
        if rid in m["exclude_ids"]:
            reason = "excluded id (test export, not in KEY_INFO)"
        elif rid not in persons:
            reason = "not in KEY_INFO"
        elif persons[rid] in m["exclude_persons"]:
            reason = f"excluded person {persons[rid]}"
        elif errors.get(rid, "-") not in ("-", "", "nan"):
            reason = f"KEY_INFO: {errors[rid]}"
        elif missing:
            reason = "missing " + ", ".join(missing)
        elif label is None:
            reason = f"label not exactly one class: {labels}"
        elif label not in m["classes"]:
            reason = f"class {label} not studied"
        rows.append({"id": rid, "person": persons.get(rid), "label": label, "include": reason is None, "reason": reason or ""})
    df = pd.DataFrame(rows)
    df["n"] = df["id"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("n").drop(columns="n").reset_index(drop=True)
