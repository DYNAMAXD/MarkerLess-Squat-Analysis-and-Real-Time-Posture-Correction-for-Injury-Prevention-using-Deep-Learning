"""
build_manifest.py   (SCRIPT, not a notebook)
=============================================
Step 1 of the squat study: copy the usable recordings and build data/manifest.csv.

What it does
------------
1. Reads KEY_INFO.xlsx (recording -> person, file errors) and the mapping CSVs (recording -> error label).
2. Decides per recording whether it is usable, and why not if it is not.
3. Copies ONLY the 6 files the models need from Temp_Dataset into data/recordings/<id>/.
   Temp_Dataset itself is never modified.
4. Writes data/manifest.csv (one row per recording).

Run it yourself (from the squat_error_study folder)
---------------------------------------------------
    C:\\squat_venv\\Scripts\\python.exe scripts\\build_manifest.py
    C:\\squat_venv\\Scripts\\python.exe scripts\\build_manifest.py --dry-run     # report only, copy nothing
    C:\\squat_venv\\Scripts\\python.exe scripts\\build_manifest.py --src D:\\other\\Temp_Dataset
"""

import argparse
import csv
import shutil
from pathlib import Path

import openpyxl

# ----------------------------------------------------------------------------
# 1. Settings
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent           # squat_error_study/
DEFAULT_SRC = Path(r"C:\Current_Mini_Project_Working\Temp_Dataset")

FILES_TO_COPY = ["points3d.trc", "advancedangles.csv", "reps.csv",
                 "rep_features.csv", "meta.json", "results.json"]

EXCLUDED_PERSONS = {"E_PANTS_MALE"}        # you asked to ignore this person
DROPPED_LABEL = "abnormal_tempo"           # class removed from the study (not a squat-form error here)
IGNORED_IDS = {"p01"}                      # test recording, ignored

# Anything not in this set is reported so you can add a mapping.
LABEL_FIX = {
    "heels_up": "heel_lift", "heel_up": "heel_lift", "heel lift": "heel_lift",
    "shallow": "insufficient_depth",
}
KNOWN_LABELS = {"correct", "insufficient_depth", "forward_lean", "knee_valgus",
                "knee_varus", "heel_lift", "abnormal_tempo"}


# ----------------------------------------------------------------------------
# 2. Read the label and person information
# ----------------------------------------------------------------------------
def clean_label(raw):
    """Strip the stray quote characters (e.g. '"correct"') and normalise the name."""
    s = raw.strip().strip('"').strip().lower().replace(" ", "_")
    return LABEL_FIX.get(s, s)


def read_labels(info_dir):
    labels = {}
    for f in sorted((info_dir / "MAPPINGS_INFO").glob("*.csv")):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                labels[row["person_id"].strip()] = clean_label(row["error_performed"])
    return labels


def read_key_info(info_dir):
    """Returns {recording_id: (person, file_error_text)}."""
    ws = openpyxl.load_workbook(info_dir / "KEY_INFO.xlsx", data_only=True).active
    out = {}
    for rid, person, err in list(ws.iter_rows(values_only=True))[1:]:
        if rid:
            out[str(rid).strip()] = (str(person).strip(), str(err or "-").strip())
    return out


def count_reps(rec_dir):
    f = rec_dir / "reps.csv"
    if not f.exists():
        return 0
    with open(f, newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


# ----------------------------------------------------------------------------
# 3. Build the manifest and copy files
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pre = args.src / "preprocessed_data"
    info = args.src / "Additional_Info"
    dest = HERE / "data" / "recordings"

    labels = read_labels(info)
    key = read_key_info(info)

    rows = []
    for rec_dir in sorted(pre.iterdir(), key=lambda p: (len(p.name), p.name)):
        rid = rec_dir.name
        if not rec_dir.is_dir() or rid in IGNORED_IDS:
            continue
        person, file_err = key.get(rid, ("?", "not in KEY_INFO"))
        label = labels.get(rid, "?")
        n_reps = count_reps(rec_dir)

        reasons = []
        if person in EXCLUDED_PERSONS:
            reasons.append("excluded person")
        if person == "?":
            reasons.append("not in KEY_INFO")
        if file_err not in ("-", ""):
            reasons.append("KEY_INFO: " + file_err.lower())
        if label == DROPPED_LABEL:
            reasons.append("abnormal_tempo class dropped (protocol not followed)")
        elif label not in KNOWN_LABELS:
            reasons.append("unknown label")
        if not (rec_dir / "reps.csv").exists():
            reasons.append("no reps.csv")
        elif n_reps < 3:
            reasons.append(f"only {n_reps} reps")
        missing = [f for f in FILES_TO_COPY if not (rec_dir / f).exists()]
        if missing and "no reps.csv" not in reasons:
            reasons.append("missing " + "/".join(missing))

        rows.append({
            "recording_id": rid, "person": person, "label": label,
            "n_reps": n_reps,
            "rep_count_note": "check (more than 12 reps)" if n_reps > 12 else "",
            "include": not reasons, "exclude_reason": "; ".join(reasons),
            "is_tempo": label == "abnormal_tempo",
        })

    # Copy only the included recordings
    n_copied = 0
    if not args.dry_run:
        for r in rows:
            if not r["include"]:
                continue
            out = dest / r["recording_id"]
            out.mkdir(parents=True, exist_ok=True)
            for f in FILES_TO_COPY:
                shutil.copy2(pre / r["recording_id"] / f, out / f)
            n_copied += 1

        man = HERE / "data" / "manifest.csv"
        man.parent.mkdir(parents=True, exist_ok=True)
        with open(man, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------------
    inc = [r for r in rows if r["include"]]
    print(f"Recordings found: {len(rows)}   included: {len(inc)}   excluded: {len(rows) - len(inc)}")
    print(f"Persons included: {len({r['person'] for r in inc})}")
    print("\nExcluded:")
    for r in rows:
        if not r["include"]:
            print(f"  {r['recording_id']:6s} {r['person']:14s} {r['exclude_reason']}")
    print("\nIncluded recordings per label:")
    for lab in sorted({r['label'] for r in inc}):
        print(f"  {lab:20s} {sum(1 for r in inc if r['label'] == lab)}")
    print("\nReps per included recording (n_reps: count):")
    from collections import Counter
    print(" ", sorted(Counter(r['n_reps'] for r in inc).items()))
    print(f"\nCopied {n_copied} recordings to {dest}" if not args.dry_run else "\n(dry run: nothing copied)")


if __name__ == "__main__":
    main()
