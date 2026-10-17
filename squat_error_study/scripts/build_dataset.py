"""
build_dataset.py   (SCRIPT)
===========================
Step 2: turn all included recordings into ONE compact file  data/processed/reps_T<T>.npz
(plus a .json describing its fields). Every model later trains from this single file.

Run it yourself (from the squat_error_study folder):
    C:\\squat_venv\\Scripts\\python.exe scripts\\build_dataset.py
    C:\\squat_venv\\Scripts\\python.exe scripts\\build_dataset.py --T 32        # other resample length
    C:\\squat_venv\\Scripts\\python.exe scripts\\build_dataset.py --limit 5     # quick test on 5 recordings
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from squat_study import features as F          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    T = args.T or cfg["data"]["T"]
    man = pd.read_csv(ROOT / cfg["paths"]["manifest"])
    man = man[man["include"]]
    if args.limit:
        man = man.head(args.limit)

    X_j, X_f, X_t, X_s, y, person, rec, rep_no, miss, tmiss = [], [], [], [], [], [], [], [], [], []
    infos, joint_names, summary_names, swapped = {}, None, None, []

    for _, row in man.iterrows():
        rdir = ROOT / cfg["paths"]["recordings"] / row["recording_id"]
        reps, info = F.recording_to_reps(rdir, T=T, min_rep_s=cfg["data"]["min_rep_s"],
                                         max_rep_s=cfg["data"]["max_rep_s"])
        joint_names = joint_names or info["joint_names"]
        summary_names = summary_names or info["summary_names"]
        if info["summary_names"] != summary_names:
            print("  WARNING: summary columns differ in", row["recording_id"], "- skipped")
            continue
        if info["lr_swapped"]:
            swapped.append(row["recording_id"])
        infos[row["recording_id"]] = {k: v for k, v in info.items()
                                      if k not in ("joint_names", "summary_names")}
        for r in reps:
            X_j.append(r["joints"]); X_f.append(r["frame"]); X_t.append(r["tempo"])
            X_s.append(r["summary"]); miss.append(r["missing"]); tmiss.append(r["trunk_missing"]); y.append(row["label"]); person.append(row["person"])
            rec.append(row["recording_id"]); rep_no.append(r["rep_no"])
        print(f"{row['recording_id']:6s} {row['label']:20s} reps kept {info['n_reps']:2d}"
              f"  dropped {info['n_dropped']}  leg {info['leg_length_m']:.2f} m"
              + ("  L/R-swapped->fixed" if info["lr_swapped"] else "")
              + (f"  MISSING {info['missing_per_joint']}" if info["missing_per_joint"] else ""))

    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"reps_T{T}.npz"
    np.savez_compressed(
        out, joints=np.stack(X_j).astype(np.float32), frame=np.stack(X_f).astype(np.float32),
        tempo=np.stack(X_t).astype(np.float32), summary=np.stack(X_s).astype(np.float32),
        label=np.array(y), person=np.array(person), recording=np.array(rec),
        rep_no=np.array(rep_no), missing=np.array(miss, dtype=np.float32),
        trunk_missing=np.array(tmiss, dtype=np.float32))
    meta = dict(T=T, n_reps=len(y), joint_names=joint_names, joint_channels=F.JOINT_CHANNELS,
                frame_feature_names=F.FRAME_FEATURE_NAMES, tempo_names=F.TEMPO_NAMES,
                summary_names=summary_names, recordings=infos,
                recordings_lr_swapped_and_fixed=swapped)
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=1)

    print(f"\nSaved {out}  ({out.stat().st_size / 1e6:.1f} MB)  reps: {len(y)}")
    print("Reps per class:", dict(pd.Series(y).value_counts()))
    m = np.array(miss)
    print(f"Left/right swapped and corrected in {len(swapped)} recordings: {swapped}")
    t = np.array(tmiss)
    print(f"Reps with a missing LOWER-BODY joint in >0% / >25% / >50% of frames: "
          f"{(m > 0).sum()} / {(m > .25).sum()} / {(m > .5).sum()}  of {len(m)}")
    print(f"Reps with the NECK missing in >0% / >25% / >50% of frames: "
          f"{(t > 0).sum()} / {(t > .25).sum()} / {(t > .5).sum()}  of {len(t)}")


if __name__ == "__main__":
    main()
