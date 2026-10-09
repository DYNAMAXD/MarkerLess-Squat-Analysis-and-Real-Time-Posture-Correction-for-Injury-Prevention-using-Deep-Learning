"""
batch_preprocess.py -- preprocess a whole two-camera (side + front) dataset into CSV files, unattended.
Runs in the Pose2Sim environment; uses the GPU when onnxruntime-gpu can (see start_batch.ps1).

Input folder
    intrinsic_front.<ext>, intrinsic_side.<ext>   board moved over the whole frame, one video per camera (required)
    extrinsic_front.<ext>, extrinsic_side.<ext>   board seen by both cameras + ONE clap (optional, see --extrinsics)
    <any name>.csv                                columns person_id, front_id, side_id, error_performed
                                                  (error_performed: comma-separated, quoted, e.g. "knee_valgus, heels_up")
    the squat videos                              file name (with or without extension) = front_id / side_id;
                                                  anywhere under the input folder
    (the calibration videos may also be folders of that name holding one video each)

--extrinsics per-video (default, recording mode 1): every squat video is  board held still, seen by both cameras
    -> clap -> squats. Each person gets their own extrinsics from that board. If extrinsic_front/side videos are
    present they are only a fallback, for a person whose board is not found (valid only if the cameras did not move).
--extrinsics shared (recording mode 2): one extrinsic calibration from extrinsic_front/side for everybody; the squat
    videos need no board.
--squat-claps 1 (default): analyse from the clap to the end; 2: the squats are between a start and an end clap.

Output folder (default <input>/preprocessed_data)
    _calibration/intrinsic/   intrinsics (computed once), log.txt
    _calibration/extrinsic/   extrinsics from extrinsic_front/side (if those videos exist): board images, log.txt
    <person_id>/              angles.csv, advancedangles.csv, reps.csv, rep_features.csv, points3d.trc,
                              calibration.toml, board_side.jpg / board_front.jpg (per-video extrinsics),
                              meta.json (ids + errors), log.txt, results.json,
                              front.mp4 / side.mp4 (540p pose overlays to check the tracking)
    _work/<person_id>/        Pose2Sim working folder (2D keypoints .json); its large videos are deleted unless --keep-work
    summary.csv               one row per person: status, frames, reps, sync, extrinsics, camera geometry, problems
    batch_status.json         live progress (read by batch_app.py)
    batch_log.txt             this script's log

Re-running the same command resumes: finished persons are skipped (--redo recomputes them). Creating the file
<output>/STOP makes the batch stop after the current person.

    python two_view/batch_preprocess.py --input D:/dataset --intr-square 41 --ext-square 41 --device auto
    python two_view/batch_preprocess.py --input D:/dataset --check     (only list what was found)
"""
import argparse
import csv
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_pose2sim.py"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".mpg", ".mpeg"}
INTR_KEYS = ("intrinsic_front", "intrinsic_side")
EXTR_KEYS = ("extrinsic_front", "extrinsic_side")
CALIB_KEYS = INTR_KEYS + EXTR_KEYS
COLUMNS = {"personid": "person_id", "person": "person_id", "frontid": "front_id", "front": "front_id",
           "sideid": "side_id", "side": "side_id", "errorperformed": "error_performed", "error": "error_performed",
           "errors": "error_performed", "errorsperformed": "error_performed"}
SKIP_DIRS = {"preprocessed_data", "_work", "_calibration"}
SUMMARY_COLS = ["person_id", "folder", "front_id", "side_id", "error_performed", "status", "seconds", "frames",
                "reps", "squat_s", "clap_offset_s", "drift_ms", "end_clap", "sync_reliable", "extrinsics",
                "ext_reproj_px", "board_px", "cams_apart_m", "view_angle_deg", "triang_err_px", "device", "problem"]
CAMS_MOVED_M = 0.10   # a person's camera distance this far from the batch median -> "cameras moved?" note


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def safe_folder(s):
    return re.sub(r'[<>:"/\\|?*\s]+', "_", str(s).strip()).strip("._") or "unnamed"


# --------------------------------------------------------------------------- discovery
def _videos(root, skip):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS and (d / n).resolve() != skip]
        out += [d / f for f in filenames if Path(f).suffix.lower() in VIDEO_EXT and not f.startswith("._")]
    return sorted(out)


def _read_csv(path):
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    head = raw.splitlines()[0] if raw.strip() else ""
    delim = ";" if head.count(";") > head.count(",") else ("\t" if head.count("\t") > head.count(",") else ",")
    rows = list(csv.reader(raw.splitlines(), delimiter=delim))
    if not rows:
        raise ValueError(f"{path} is empty")
    cols = [COLUMNS.get(norm(h), norm(h)) for h in rows[0]]
    missing = [c for c in ("person_id", "front_id", "side_id") if c not in cols]
    if missing:
        raise ValueError(f"{Path(path).name}: missing column(s) {missing}; found {rows[0]}. Expected person_id, "
                         "front_id, side_id, error_performed.")
    out = []
    for r in rows[1:]:
        if not any(x.strip() for x in r):
            continue
        rec = {c: (r[i].strip() if i < len(r) else "") for i, c in enumerate(cols)}
        rec.setdefault("error_performed", "")
        extra = [x.strip() for x in r[len(cols):] if x.strip()]  # unquoted commas spill into extra columns
        if extra:
            rec["error_performed"] = ", ".join([rec["error_performed"]] + extra).strip(", ")
        out.append(rec)
    return out


def discover(input_dir, output_dir=None, csv_path=None, extrinsics="per-video"):
    """What the input folder holds -> plan dict (also used by batch_app.py for the 'Scan' button)."""
    root = Path(input_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else root / "preprocessed_data"
    plan = {"input": str(root), "output": str(out_dir), "extrinsics": extrinsics, "calib": {}, "csv": None,
            "rows": [], "errors": [], "warnings": []}
    if not root.is_dir():
        plan["errors"].append(f"Input folder not found: {root}")
        return plan
    vids = _videos(root, out_dir)
    by_name, by_stem = {}, {}
    for v in vids:
        by_name.setdefault(v.name.lower(), []).append(v)
        by_stem.setdefault(v.stem.lower(), []).append(v)

    for key in CALIB_KEYS:
        hits = [v for v in vids if norm(v.stem) == norm(key)] or [v for v in vids if norm(v.parent.name) == norm(key)]
        if len(hits) > 1:  # prefer the one at the top of the input folder
            top = [v for v in hits if v.parent == root]
            hits = top if len(top) == 1 else hits
        needed = key in INTR_KEYS or extrinsics == "shared"
        if not hits:
            if needed:
                plan["errors"].append(f"No {key} video (expected a file named {key}.mp4 or a folder {key}/ with one video).")
        elif len(hits) > 1:
            (plan["errors"] if needed else plan["warnings"]).append(
                f"Several {key} videos: {', '.join(str(h.relative_to(root)) for h in hits)}")
        plan["calib"][key] = str(hits[0]) if len(hits) == 1 else None
    have_ext = all(plan["calib"].get(k) for k in EXTR_KEYS)
    if extrinsics == "per-video":
        plan["warnings"].append(
            "Extrinsics from the board at the start of each squat video; extrinsic_front/side found -> used only for "
            "a person whose board is not found." if have_ext else
            "Extrinsics from the board at the start of each squat video (no extrinsic_front/side fallback videos).")

    if csv_path:
        csvs = [Path(csv_path)]
    else:
        csvs = sorted(p for p in root.glob("*.csv"))
        if len(csvs) > 1:
            with_ids = [p for p in csvs if "personid" in norm(p.read_text(encoding="utf-8-sig", errors="replace")[:300])]
            csvs = with_ids or csvs
    if not csvs:
        plan["errors"].append("No .csv mapping file in the input folder (columns person_id, front_id, side_id, "
                              "error_performed).")
        return plan
    if len(csvs) > 1:
        plan["errors"].append(f"Several mapping CSVs: {', '.join(p.name for p in csvs)}. Keep one or pick it.")
        return plan
    plan["csv"] = str(csvs[0])
    try:
        recs = _read_csv(csvs[0])
    except Exception as e:  # noqa: BLE001
        plan["errors"].append(str(e))
        return plan

    calib_files = {Path(p) for p in plan["calib"].values() if p}
    seen = {}
    for n, rec in enumerate(recs, start=2):  # line number in the CSV (header = 1)
        row = {"line": n, "person_id": rec.get("person_id", ""), "front_id": rec.get("front_id", ""),
               "side_id": rec.get("side_id", ""), "error_performed": rec.get("error_performed", ""), "problems": []}
        row["errors"] = [e.strip() for e in row["error_performed"].split(",") if e.strip()]
        if not row["person_id"]:
            row["problems"].append("empty person_id")
        folder = safe_folder(row["person_id"])
        if folder.lower() in seen:
            seen[folder.lower()] += 1
            plan["warnings"].append(f"person_id {row['person_id']} appears more than once (CSV line {n}); "
                                    f"saved as {folder}_{seen[folder.lower()]}")
            folder = f"{folder}_{seen[folder.lower()]}"
        else:
            seen[folder.lower()] = 1
        row["folder"] = folder
        for cam in ("front", "side"):
            vid_id = row[f"{cam}_id"]
            hits = by_name.get(vid_id.lower()) or by_stem.get(vid_id.lower()) or by_stem.get(Path(vid_id).stem.lower()) or []
            hits = [h for h in hits if h not in calib_files]
            if not vid_id:
                row["problems"].append(f"empty {cam}_id")
            elif not hits:
                row["problems"].append(f"{cam} video '{vid_id}' not found")
            elif len(hits) > 1:
                row["problems"].append(f"{cam} video '{vid_id}' is ambiguous: "
                                       + ", ".join(str(h.relative_to(root)) for h in hits))
            row[cam] = str(hits[0]) if len(hits) == 1 else None
        if row["front"] and row["front"] == row["side"]:
            row["problems"].append("front and side are the same file")
        plan["rows"].append(row)
    if not plan["rows"]:
        plan["errors"].append(f"{csvs[0].name} has no data rows.")
    bad = sum(1 for r in plan["rows"] if r["problems"])
    if bad:
        plan["warnings"].append(f"{bad} of {len(plan['rows'])} CSV rows have problems and will be skipped.")
    return plan


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(fps, 3), "dur": round(n / fps, 1) if fps else 0.0, "ok": cap.isOpened()}
    cap.release()
    return info


# --------------------------------------------------------------------------- helpers
class Batch:
    def __init__(self, a, plan):
        self.a, self.plan = a, plan
        self.out = Path(plan["output"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_fh = open(self.out / "batch_log.txt", "a", encoding="utf-8")
        self.status = {"state": "starting", "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "t0": time.time(), "input": plan["input"], "output": plan["output"], "csv": plan["csv"],
                       "settings": {k: v for k, v in vars(a).items() if k not in ("input", "output", "csv")},
                       "device": None, "calib": {"state": "pending"}, "current": None, "persons": []}
        for r in plan["rows"]:
            self.status["persons"].append({
                "person_id": r["person_id"], "folder": r["folder"], "front_id": r["front_id"], "side_id": r["side_id"],
                "error_performed": r["error_performed"], "state": "invalid" if r["problems"] else "pending",
                "problem": "; ".join(r["problems"]), "stage": None, "seconds": None})

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        self.log_fh.write(line + "\n")
        self.log_fh.flush()

    def save(self):
        self.status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.out / "batch_status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=1), encoding="utf-8")
        for _ in range(5):  # a reader may hold the file open for a moment on Windows
            try:
                os.replace(tmp, self.out / "batch_status.json")
                return
            except PermissionError:
                time.sleep(0.2)

    def write_summary(self):
        rows = []
        for p in self.status["persons"]:
            r = {c: "" if p.get(c) is None else p.get(c) for c in SUMMARY_COLS}
            r["status"] = p["state"]
            rows.append(r)
        try:
            with open(self.out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, SUMMARY_COLS)
                w.writeheader()
                w.writerows(rows)
        except PermissionError:  # open in Excel: try again after the next person
            self.log("note: summary.csv is open in another program; not updated this time")


def keep_awake(on=True):
    """Stop Windows from sleeping while the batch runs (no power settings are changed)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def read_log(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_stage(log_path):
    stage = None
    for line in read_log(log_path).splitlines():
        if line.startswith("[stage] "):
            stage = line[8:].strip()
    return stage


def error_line(log_path):
    """The most useful last line of a failed run's log (the exception message)."""
    lines = [ln.strip() for ln in read_log(log_path).splitlines() if ln.strip()]
    if not lines:
        return "no log written"
    for ln in reversed(lines):
        if re.match(r"^[A-Za-z0-9_.]*(Error|Exception|BoardNotFound)\b", ln):
            return ln[:400]
    return lines[-1][:400]


def board_not_found(log_path):
    return "BoardNotFound" in read_log(log_path)


def run_child(cmd, console, log_path, timeout_s, on_tick):
    """Runs one run_pose2sim.py process; calls on_tick(stage) every few seconds. Returns (exit code, timed out)."""
    with open(console, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent))
        t0 = time.time()
        try:
            while proc.poll() is None:
                time.sleep(3)
                on_tick(last_stage(log_path))
                if time.time() - t0 > timeout_s:
                    kill_tree(proc)
                    return -1, True
        except BaseException:
            kill_tree(proc)
            raise
    return proc.returncode, False


def board_args(a):
    return ["--intr-corners", a.intr_corners, "--intr-square", str(a.intr_square), "--ext-corners", a.ext_corners,
            "--ext-square", str(a.ext_square), "--board-position", a.board_position, "--clap-window", str(a.clap_window)]


def extrinsic_info(log):
    """Extrinsic quality lines that run_pose2sim / p2s_extrinsics print -> dict."""
    out = {}
    m = re.findall(r"extrinsics: reprojection error side ([\d.]+) px, front ([\d.]+) px", log)
    if m:
        out["ext_reproj_px"] = max(float(m[-1][0]), float(m[-1][1]))
    m = re.findall(r"using the one where it is largest: .*\(~(\d+) px\)", log)
    if m:
        out["board_px"] = int(m[-1])
    m = re.findall(r"cameras are ([\d.]+) m apart; angle between their viewing directions (\d+) deg", log)
    if m:
        out["cams_apart_m"], out["view_angle_deg"] = float(m[-1][0]), int(m[-1][1])
    return out


# --------------------------------------------------------------------------- calibration (once)
def _calib_step(b, name, cmd_tail, settings, what):
    """Runs one run_pose2sim calibration step into _calibration/<name>/ unless an identical one is there."""
    cdir = b.out / "_calibration" / name
    stamp = cdir / "settings.json"
    if (cdir / "results.json").exists() and stamp.exists() and not b.a.redo_calibration and \
            json.loads(stamp.read_text(encoding="utf-8")) == settings:
        b.log(f"calibration: reusing _calibration/{name}/ (same videos and board settings)")
        return cdir
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    (b.out / "_work").mkdir(exist_ok=True)
    b.status["calib"].update(state="running", step=what, stage="setup")
    b.save()
    b.log(f"calibration: {what} ...")
    cmd = [sys.executable, "-u", str(RUN), "--name", f"calibration_{name}", *cmd_tail, *board_args(b.a),
           "--out-dir", str(cdir), "--work-dir", str(b.out / "_work" / f"_{name}")]

    def tick(stage):
        b.status["calib"]["stage"] = stage
        b.save()
    t0 = time.time()
    code, timed_out = run_child(cmd, b.out / "_work" / f"calibration_{name}_console.txt", cdir / "log.txt", 3600, tick)
    if code != 0 or not (cdir / "results.json").exists():
        msg = "timed out after 1 h" if timed_out else error_line(cdir / "log.txt")
        raise RuntimeError(f"{what} failed: {msg}  (full log: {cdir / 'log.txt'})")
    stamp.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    b.log(f"calibration: {what} done in {time.time() - t0:.0f} s")
    return cdir


def do_calibration(b):
    """Intrinsics once (+ shared extrinsics from extrinsic_front/side when present).
    Returns (intrinsics toml, shared/fallback calibration toml or None)."""
    import rtoml
    a, plan = b.a, b.plan
    b.status["calib"] = {"state": "running"}
    intr_set = {k: getattr(a, k) for k in ("intr_corners", "intr_square")}
    intr_set["videos"] = {k: plan["calib"][k] for k in INTR_KEYS}
    try:
        idir = _calib_step(b, "intrinsic", ["--mode", "intrinsic", "--intr-side", plan["calib"]["intrinsic_side"],
                                            "--intr-front", plan["calib"]["intrinsic_front"]], intr_set,
                           "intrinsics from intrinsic_front/side")
    except RuntimeError as e:
        b.status["calib"] = {"state": "error", "problem": str(e)}
        b.save()
        raise
    rep = {"cams": {}, "warnings": [], "extrinsics": None, "boards": []}
    data = rtoml.load(idir / "calibration.toml")
    rms = [float(x) for x in re.findall(r"Intrinsics error: ([\d.]+) px", read_log(idir / "log.txt"))]
    for k, (name, c) in enumerate((n, c) for n, c in sorted(data.items()) if isinstance(c, dict) and "matrix" in c):
        K, (w, h) = c["matrix"], c["size"]
        cam = {"fx": round(K[0][0], 1), "fy": round(K[1][1], 1), "cx": round(K[0][2], 1), "cy": round(K[1][2], 1),
               "size": f"{int(w)}x{int(h)}", "intr_rms_px": rms[k] if k < len(rms) else None}
        rep["cams"][name] = cam
        b.log(f"calibration: {name} fx {cam['fx']} fy {cam['fy']} cx {cam['cx']} cy {cam['cy']} RMS {cam['intr_rms_px']} px")
        if abs(cam["fx"] / cam["fy"] - 1) > 0.03:
            rep["warnings"].append(f"{name}: fx {cam['fx']} and fy {cam['fy']} differ by more than 3 %")
        if abs(cam["cx"] - w / 2) > 0.05 * w or abs(cam["cy"] - h / 2) > 0.06 * h:
            rep["warnings"].append(f"{name}: principal point ({cam['cx']}, {cam['cy']}) is far from the image centre")
        if cam["intr_rms_px"] and cam["intr_rms_px"] > 2:
            rep["warnings"].append(f"{name}: intrinsic RMS {cam['intr_rms_px']} px is high")

    shared = None
    if all(plan["calib"].get(k) for k in EXTR_KEYS):
        ext_set = {"intrinsics": intr_set, **{k: getattr(a, k) for k in ("ext_corners", "ext_square", "board_position")},
                   "videos": {k: plan["calib"][k] for k in EXTR_KEYS}}
        try:
            edir = _calib_step(b, "extrinsic", ["--mode", "extrinsic", "--side", plan["calib"]["extrinsic_side"],
                                                "--front", plan["calib"]["extrinsic_front"],
                                                "--intr-from", str(idir / "calibration.toml")], ext_set,
                               "extrinsics from extrinsic_front/side")
            shared = edir / "calibration.toml"
            rep["extrinsics"] = extrinsic_info(read_log(edir / "log.txt"))
            rep["boards"] = [f"_calibration/extrinsic/{f}" for f in ("board_side.jpg", "board_front.jpg")
                             if (edir / f).exists()]
            rep["warnings"] += re.findall(r"\[calibration\] WARNING: (.*)", read_log(edir / "log.txt"))
        except RuntimeError as e:
            if a.extrinsics == "shared":
                b.status["calib"] = {"state": "error", "problem": str(e)}
                b.save()
                raise
            rep["warnings"].append(f"No fallback extrinsics: {e}")
            b.log(f"calibration: WARNING {e}")
    rep["mode"] = a.extrinsics
    b.status["calib"] = {"state": "done", **rep}
    for w in rep["warnings"]:
        b.log(f"calibration: WARNING {w}")
    b.save()
    return idir / "calibration.toml", shared


# --------------------------------------------------------------------------- one person
def person_result(pdir, p):
    res = json.loads((pdir / "results.json").read_text(encoding="utf-8"))
    sy = res.get("sync", {})
    p.update(frames=res.get("counts", {}).get("frames"), reps=res.get("counts", {}).get("reps"),
             squat_s=sy.get("squat_dur_s"), clap_offset_s=sy.get("offset_s"),
             drift_ms=None if sy.get("drift_s") is None else round(1000 * sy["drift_s"]),
             end_clap=None if "t_end_side_s" not in sy else ("yes" if sy["t_end_side_s"] else "no"),
             sync_reliable=sy.get("reliable"), device=res.get("device"))
    log = read_log(pdir / "log.txt")
    p.update(extrinsic_info(log))
    m = re.findall(r"Mean reprojection error for all points on frames \d+ to \d+ is ([\d.]+) px", log)
    p["triang_err_px"] = float(m[-1]) if m else None
    notes = [p["problem"]] if p.get("problem") else []
    if not p["reps"]:
        notes.append("no squat reps detected")
    if p["end_clap"] == "no":
        notes.append("no end clap: analysed to the end of the video")
    if p["sync_reliable"] is False:
        notes.append("clap sync uncertain")
    if p.get("ext_reproj_px") and p["ext_reproj_px"] > 1.0:
        notes.append(f"extrinsic reprojection {p['ext_reproj_px']} px")
    if p.get("board_px") and p["board_px"] < 150:
        notes.append(f"board small ({p['board_px']} px)")
    if p["triang_err_px"] and p["triang_err_px"] > 15:
        notes.append(f"high triangulation error {p['triang_err_px']} px")
    p["problem"] = "; ".join(notes)


def do_person(b, row, p, intr_toml, shared_toml):
    a = b.a
    pdir, work = b.out / row["folder"], b.out / "_work" / row["folder"]
    if (pdir / "results.json").exists() and not a.redo:
        p.update(state="done", extrinsics=p.get("extrinsics") or "(earlier run)")
        try:
            meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
            p["extrinsics"] = meta.get("extrinsics", p["extrinsics"])
            person_result(pdir, p)
        except Exception:  # noqa: BLE001
            pass
        p["problem"] = ("already done (skipped); " + (p.get("problem") or "")).strip("; ")
        return
    attempts = [("shared", ["--calib-from", str(shared_toml)])] if a.extrinsics == "shared" else \
        [("own board", ["--intr-from", str(intr_toml)])] + \
        ([("fallback: extrinsic videos", ["--calib-from", str(shared_toml)])] if shared_toml else [])
    t0 = time.time()
    for k, (label, calib_args) in enumerate(attempts):
        if pdir.exists():
            shutil.rmtree(pdir)
        pdir.mkdir(parents=True)
        work.mkdir(parents=True, exist_ok=True)
        (pdir / "meta.json").write_text(json.dumps({
            "name": row["folder"], "kind": "pose2sim", "person_id": row["person_id"], "subject": row["person_id"],
            "front_id": row["front_id"], "side_id": row["side_id"], "error_performed": row["error_performed"],
            "errors": row["errors"], "label": row["error_performed"], "front_file": row["front"],
            "side_file": row["side"], "extrinsics": label, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1),
            encoding="utf-8")
        cmd = [sys.executable, "-u", str(RUN), "--name", row["folder"], "--side", row["side"], "--front", row["front"],
               *calib_args, "--squat-claps", str(a.squat_claps), *board_args(a), "--device", a.device,
               "--pose-mode", a.pose_mode, "--out-dir", str(pdir), "--work-dir", str(work)]
        p.update(state="running", stage="setup", started=time.strftime("%H:%M:%S"), extrinsics=label)
        b.status["current"] = row["folder"]
        b.save()

        def tick(stage):
            p["stage"], p["seconds"] = stage, round(time.time() - t0)
            b.save()
        code, timed_out = run_child(cmd, work / "console.txt", pdir / "log.txt", a.timeout_min * 60, tick)
        p["seconds"] = round(time.time() - t0)
        if code == 0 and (pdir / "results.json").exists():
            p.update(state="done", problem="")
            if k > 0:
                p["problem"] = "board not found in this person's videos: used the extrinsic_front/side calibration"
            person_result(pdir, p)
            if not a.keep_work:  # the cut videos and full-size overlays are large; the 540p overlays are in pdir
                for f in list((work / "trial" / "videos").glob("*.mp4")) + list((work / "trial" / "pose").glob("*.mp4")):
                    f.unlink(missing_ok=True)
            break
        p["state"] = "error"
        p["problem"] = f"timed out after {a.timeout_min} min" if timed_out else error_line(pdir / "log.txt")
        if not (k + 1 < len(attempts) and board_not_found(pdir / "log.txt")):
            break
        shutil.copyfile(pdir / "log.txt", work / "log_own_board_failed.txt")
        b.log(f"{p['person_id']}: board not found ({p['problem']}); retrying with the extrinsic_front/side calibration")
    p["stage"] = None


def flag_moved_cameras(b):
    """Camera distance far from the batch median -> note on that person (cameras moved, or a bad board frame)."""
    vals = sorted(p["cams_apart_m"] for p in b.status["persons"] if p["state"] == "done" and p.get("cams_apart_m"))
    if len(vals) < 3:
        return
    med = vals[len(vals) // 2]
    b.status["cams_apart_median_m"] = med
    for p in b.status["persons"]:
        d = p.get("cams_apart_m")
        note = f"camera distance {d} m vs median {med} m: cameras moved or bad board frame?"
        if p["state"] == "done" and d and abs(d - med) > CAMS_MOVED_M and "camera distance" not in (p.get("problem") or ""):
            p["problem"] = "; ".join(x for x in (p.get("problem"), note) if x)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="dataset folder (calibration videos, mapping CSV, squat videos)")
    ap.add_argument("--output", help="output folder (default <input>/preprocessed_data)")
    ap.add_argument("--csv", help="mapping CSV (default: the one .csv in the input folder)")
    ap.add_argument("--extrinsics", default="per-video", choices=["per-video", "shared"],
                    help="per-video = board at the start of every squat video (mode 1); "
                         "shared = one calibration from extrinsic_front/side for everybody (mode 2)")
    ap.add_argument("--intr-corners", default="6,7", help="intrinsic board INNER corners, e.g. 6,7")
    ap.add_argument("--intr-square", type=float, default=41.0, help="intrinsic board square size, mm")
    ap.add_argument("--ext-corners", default="6,7", help="extrinsic board INNER corners")
    ap.add_argument("--ext-square", type=float, default=41.0, help="extrinsic board square size, mm")
    ap.add_argument("--board-position", default="vertical", choices=["vertical", "horizontal"],
                    help="extrinsic board held/standing (vertical) or flat on the floor (horizontal)")
    ap.add_argument("--clap-window", type=float, default=90.0, help="one-clap videos: clap within the first N s")
    ap.add_argument("--squat-claps", type=int, default=1, choices=[1, 2],
                    help="claps in each squat video: 1 = clap then squats to the end, 2 = squats between two claps")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--pose-mode", default="performance", choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--timeout-min", type=float, default=90.0, help="give up on one person after this long")
    ap.add_argument("--redo", action="store_true", help="recompute persons that are already done")
    ap.add_argument("--redo-calibration", action="store_true", help="recompute the calibration")
    ap.add_argument("--keep-work", action="store_true", help="keep the large intermediate videos in _work/")
    ap.add_argument("--check", action="store_true", help="only list what was found in the input folder")
    a = ap.parse_args()

    plan = discover(a.input, a.output, a.csv, a.extrinsics)
    if a.check:
        print(json.dumps({k: v for k, v in plan.items() if k != "rows"}, indent=1))
        for r in plan["rows"]:
            print(f"  {r['person_id']:>12}  front {r['front_id']:>14} side {r['side_id']:>14}  errors {r['errors']}  "
                  + ("OK" if not r["problems"] else "PROBLEM: " + "; ".join(r["problems"])))
        sys.exit(1 if plan["errors"] else 0)
    if plan["errors"]:
        print("Cannot start:\n  " + "\n  ".join(plan["errors"]), file=sys.stderr)
        sys.exit(2)

    b = Batch(a, plan)
    stop_file = b.out / "STOP"
    stop_file.unlink(missing_ok=True)
    keep_awake(True)
    try:
        b.log(f"batch started: {len(plan['rows'])} persons from {plan['csv']} -> {b.out}; extrinsics {a.extrinsics}, "
              f"{a.squat_claps} clap(s) per squat video, board {a.ext_corners} @ {a.ext_square} mm ({a.board_position})")
        for w in plan["warnings"]:
            b.log(f"note: {w}")
        sys.path.insert(0, str(HERE))
        from run_pose2sim import resolve_device
        b.status["device"] = resolve_device(a.device)
        b.log(f"pose estimation device: {b.status['device']} (asked: {a.device}), RTMPose mode {a.pose_mode}")
        a.device = b.status["device"]  # every person uses the same device
        b.status["state"] = "calibrating"
        b.save()
        intr_toml, shared_toml = do_calibration(b)
        b.status["state"] = "running"
        rows = {r["folder"]: r for r in plan["rows"]}
        for p in b.status["persons"]:
            if stop_file.exists():
                b.log("STOP file found: stopping before the next person")
                b.status["state"] = "stopped"
                break
            if p["state"] == "invalid":
                b.log(f"{p['person_id']}: skipped ({p['problem']})")
                continue
            b.log(f"{p['person_id']}: front {p['front_id']}, side {p['side_id']}, errors '{p['error_performed']}' ...")
            try:
                do_person(b, rows[p["folder"]], p, intr_toml, shared_toml)
            except Exception:  # noqa: BLE001  never let one person stop the batch
                p.update(state="error", problem=traceback.format_exc().strip().splitlines()[-1][:400], stage=None)
            flag_moved_cameras(b)
            b.log(f"{p['person_id']}: {p['state']} in {p.get('seconds') or 0} s"
                  + (f"  ({p['problem']})" if p.get("problem") else ""))
            b.status["current"] = None
            b.save()
            b.write_summary()
        else:
            b.status["state"] = "finished"
        n = {s: sum(1 for p in b.status["persons"] if p["state"] == s) for s in ("done", "error", "invalid", "pending")}
        b.log(f"batch {b.status['state']}: {n['done']} done, {n['error']} failed, {n['invalid']} invalid CSV rows, "
              f"{n['pending']} not started. Summary: {b.out / 'summary.csv'}")
    except BaseException as e:
        b.status["state"] = "error"
        b.status["problem"] = str(e) or type(e).__name__
        b.log("batch stopped by an error:\n" + traceback.format_exc())
        raise
    finally:
        b.status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        b.save()
        b.write_summary()
        keep_awake(False)


if __name__ == "__main__":
    main()
