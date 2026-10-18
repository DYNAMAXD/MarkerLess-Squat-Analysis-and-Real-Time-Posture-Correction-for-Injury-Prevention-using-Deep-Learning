"""
app.py -- Pose2Sim Web: a local website to process a two-camera (side + front) recording with Pose2Sim and look at the
joint angles. No rep detection is used by the website.

    C:\\Pose2SimVenv\\Scripts\\python.exe pose2sim_web\\app.py            then open  http://127.0.0.1:5100
    options:  --port 5100   --data <folder where every run is stored>   (default: pose2sim_web\\data)

What it does
  * the form takes the intrinsic videos (or a saved intrinsics file), the extrinsic board (inside the squat videos,
    or separate extrinsic videos), the side + front videos, board sizes, and the pose-model quality;
  * the processing engine is the existing, tested two_view\\run_pose2sim.py (clap sync, calibration, board detection,
    RTMPose, triangulation). This site only starts it and stores everything;
  * every run is saved in its own folder under data\\<run>\\ :
        settings.json  run.log  results\\ (points3d.trc, advancedangles.csv, calibration.toml, board pictures, log)
        processed_videos\\ (synced front/side + pose overlays)   angles_all.csv  angles_meta.json  kinovea\\
  * the viewer shows any selected angles (left and right) as a graph; touching the graph shows the values.
"""
import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENGINE = REPO / "two_view" / "run_pose2sim.py"
sys.path.insert(0, str(HERE))
import postprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5100)
ap.add_argument("--data", default=str(HERE / "data"))
ARGS, _ = ap.parse_known_args()
DATA = Path(ARGS.data).resolve()
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "_system").mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 ** 3
NAME_OK = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
STAGES = ["setup", "sync", "calibration", "extract", "pose", "triangulation", "angles", "packaging"]
SERVE_TOP = {"results", "processed_videos", "kinovea", "input"}
SERVE_FILES = {"run.log", "settings.json", "angles_all.csv", "angles_meta.json"}
PROCS = {}          # run name -> {"proc": Popen or None, "t0": float}
MODEL_JOBS = {}     # pose mode -> Popen


def slug(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", (s or "").strip()).strip("_")[:50]


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- system check
def model_files():
    """{mode: [cached .onnx file names]} the Body_with_feet models Pose2Sim needs, from rtmlib itself."""
    try:
        from rtmlib import BodyWithFeet
        out = {}
        for mode, d in BodyWithFeet.MODE.items():
            out[mode] = [Path(d[k]).name.split(".")[0] + ".onnx" for k in ("det", "pose")]
        return out
    except Exception:  # noqa: BLE001
        return {}


def cache_dir():
    return Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"


def internet_ok():
    try:
        socket.getaddrinfo("download.openmmlab.com", 443)
    except OSError as e:
        return False, f"cannot find download.openmmlab.com ({e})"
    try:
        urllib.request.urlopen(urllib.request.Request("https://download.openmmlab.com", method="HEAD"), timeout=8)
    except urllib.error.HTTPError:
        return True, "reachable"           # any HTTP answer (even 403) means the host is reachable
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return True, "reachable"


@app.get("/api/system")
def system():
    info = {"python": sys.version.split()[0], "executable": sys.executable, "engine": ENGINE.is_file(), "data_dir": str(DATA)}
    try:
        import Pose2Sim
        info["pose2sim"] = getattr(Pose2Sim, "__version__", "installed")
    except Exception as e:  # noqa: BLE001
        info["pose2sim"] = None
        info["pose2sim_error"] = str(e)
    try:
        import onnxruntime as ort
        info["onnx_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        info["onnx_providers"] = []
    ok, msg = internet_ok()
    info["internet"], info["internet_msg"] = ok, msg
    cached = {f.name for f in cache_dir().glob("*.onnx")} if cache_dir().exists() else set()
    info["models"] = {mode: {"files": fs, "ready": all(f in cached for f in fs),
                             "downloading": mode in MODEL_JOBS and MODEL_JOBS[mode].poll() is None}
                      for mode, fs in model_files().items()}
    info["disk_free_gb"] = round(shutil.disk_usage(DATA).free / 1e9, 1)
    return jsonify(info)


@app.post("/api/models/<mode>")
def download_models(mode):
    if mode not in ("lightweight", "balanced", "performance"):
        abort(404)
    if mode in MODEL_JOBS and MODEL_JOBS[mode].poll() is None:
        return jsonify({"ok": True, "note": "already downloading"})
    code = ("from rtmlib import BodyWithFeet; "
            f"BodyWithFeet(mode='{mode}', backend='onnxruntime', device='cpu'); print('MODELS READY')")
    log = open(DATA / "_system" / f"models_{mode}.log", "w", encoding="utf-8")
    MODEL_JOBS[mode] = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=log, stderr=subprocess.STDOUT)
    return jsonify({"ok": True})


@app.get("/api/models/<mode>/log")
def models_log(mode):
    p = DATA / "_system" / f"models_{mode}.log"
    txt = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    txt = "\n".join(ln.split("\r")[-1] for ln in txt.split("\n"))
    return jsonify({"log": "\n".join(txt.splitlines()[-12:]), "running": mode in MODEL_JOBS and MODEL_JOBS[mode].poll() is None})


# ----------------------------------------------------------------------------------------------- creating a run
def save_upload(field, dest_dir, stem):
    f = request.files.get(field)
    if f and f.filename:
        dest = Path(dest_dir) / f"{stem}{Path(f.filename).suffix.lower() or '.mp4'}"
        f.save(dest)
        return dest
    p = request.form.get(f"{field}_path", "").strip().strip('"').strip("'")
    if p:
        if not Path(p).is_file():
            raise ValueError(f"File not found: {p}")
        return Path(p)
    return None


def corners_ok(txt, what):
    nums = re.findall(r"\d+", txt or "")
    if len(nums) != 2 or min(map(int, nums)) < 2:
        raise ValueError(f"{what}: give the INNER corners as two numbers, e.g. 6,7 (a board of 7 x 8 squares has 6 x 7 inner corners).")
    return f"{nums[0]},{nums[1]}"


def positive(txt, what):
    try:
        v = float(txt)
        if v <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{what}: give the square size in millimetres, e.g. 41.") from None
    return str(v)


@app.post("/api/runs")
def create_run():
    f = request.form
    name = slug(f.get("name")) or "run"
    name = f"{name}_{time.strftime('%m%d-%H%M%S')}"
    run = DATA / name
    try:
        if run.exists():
            raise ValueError("A run with this name already exists, try again.")
        run.mkdir()
        (run / "input").mkdir()
        S = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        S["intr_corners"] = corners_ok(f.get("intr_corners", "6,7"), "Intrinsic board corners")
        S["intr_square"] = positive(f.get("intr_square"), "Intrinsic square size")
        S["ext_corners"] = corners_ok(f.get("ext_corners", "6,7"), "Extrinsic board corners")
        S["ext_square"] = positive(f.get("ext_square"), "Extrinsic square size")
        S["board_position"] = "horizontal" if f.get("board_position") == "horizontal" else "vertical"
        S["squat_claps"] = "2" if f.get("squat_claps") == "2" else "1"
        S["clap_window"] = str(float(f.get("clap_window") or 90))
        S["pose_mode"] = f.get("pose_mode") if f.get("pose_mode") in ("lightweight", "balanced", "performance") else "balanced"
        S["device"] = f.get("device") if f.get("device") in ("auto", "cpu", "cuda") else "cpu"
        S["intr_mode"] = "reuse" if f.get("intr_mode") == "reuse" else "videos"
        S["ext_mode"] = "separate" if f.get("ext_mode") == "separate" else "in_video"
        V = {}
        for fld in ("side", "front") + (("intr_side", "intr_front") if S["intr_mode"] == "videos" else ()) + \
                   (("ext_side", "ext_front") if S["ext_mode"] == "separate" else ()):
            V[fld] = save_upload(fld, run / "input", fld)
            if V[fld] is None:
                raise ValueError(f"Please choose the video for: {fld.replace('_', ' ')}.")
        intr_from = None
        if S["intr_mode"] == "reuse":
            src = f.get("intr_from_run", "")
            if src and src != "path":
                intr_from = DATA / src / "results" / "calibration.toml"
            else:
                intr_from = Path(f.get("intr_from_path", "").strip().strip('"'))
            if not intr_from.is_file():
                raise ValueError(f"Saved intrinsics file not found: {intr_from}")
            S["intr_from"] = str(intr_from)
        S["files"] = {k: str(v) for k, v in V.items()}
    except ValueError as e:
        shutil.rmtree(run, ignore_errors=True)
        return jsonify({"error": str(e)}), 400
    write_json(run / "settings.json", S)

    py = sys.executable
    base = [py, "-u", str(ENGINE), "--device", S["device"], "--pose-mode", S["pose_mode"], "--clap-window", S["clap_window"],
            "--ext-corners", S["ext_corners"], "--ext-square", S["ext_square"], "--intr-corners", S["intr_corners"],
            "--intr-square", S["intr_square"], "--board-position", S["board_position"]]
    intr_args = (["--intr-side", str(V["intr_side"]), "--intr-front", str(V["intr_front"])] if intr_from is None
                 else ["--intr-from", str(intr_from)])
    steps = []
    if S["ext_mode"] == "separate":
        steps.append(("extrinsic calibration from the separate board videos",
                      base + ["--mode", "extrinsic", "--name", name + "_ext", "--side", str(V["ext_side"]), "--front", str(V["ext_front"]),
                              "--out-dir", str(run / "calibration_extrinsic"), "--work-dir", str(run / "p2s_ext")] + intr_args))
        steps.append(("pose + 3D on the squat videos",
                      base + ["--name", name, "--side", str(V["side"]), "--front", str(V["front"]), "--squat-claps", S["squat_claps"],
                              "--calib-from", str(run / "calibration_extrinsic" / "calibration.toml"),
                              "--out-dir", str(run / "results"), "--work-dir", str(run / "pose2sim_project")]))
    else:
        steps.append(("calibration, pose + 3D",
                      base + ["--name", name, "--side", str(V["side"]), "--front", str(V["front"]), "--squat-claps", S["squat_claps"],
                              "--out-dir", str(run / "results"), "--work-dir", str(run / "pose2sim_project")] + intr_args))
    write_json(run / "state.json", {"status": "queued", "created": S["created"]})
    threading.Thread(target=run_job, args=(name, steps), daemon=True).start()
    return jsonify({"name": name})


def finished_before_crash(run):
    """The engine can crash while exiting (onnxruntime/multiprocessing on Windows) after everything is written."""
    try:
        text = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "[stage] done" in text and (run / "results" / "results.json").exists()


def run_job(name, steps):
    run = DATA / name
    st = {"status": "running", "created": read_json(run / "settings.json", {}).get("created"), "started": time.time()}
    write_json(run / "state.json", st)
    t0 = time.time()
    with open(run / "run.log", "w", encoding="utf-8") as log:
        for label, cmd in steps:
            log.write(f"\n[step] {label}\n$ {' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
            PROCS[name] = {"proc": proc, "t0": t0}
            code = proc.wait()
            if code != 0 and finished_before_crash(run):
                log.write(f"
[note] step '{label}' returned code {code} AFTER writing all results (Windows crash on exit, "
                          "harmless); continuing.
")
                code = 0
            if code != 0:
                log.write(f"\n[error] step '{label}' ended with code {code}. See the lines above.\n")
                st.update(status="error", finished=time.time())
                write_json(run / "state.json", st)
                return
        try:
            n = postprocess.build(run)
            vids = postprocess.collect_videos(run)
            log.write(f"\n[viewer] wrote angles_all.csv ({n} frames); processed videos saved: {', '.join(vids) or 'none'}\n")
        except Exception as e:  # noqa: BLE001
            log.write(f"\n[error] the engine finished but preparing the viewer data failed: {type(e).__name__}: {e}\n")
            st.update(status="error", finished=time.time())
            write_json(run / "state.json", st)
            return
        log.write(f"\n[finished] total {time.time() - t0:.0f} s\n")
    st.update(status="done", finished=time.time())
    write_json(run / "state.json", st)
    PROCS.pop(name, None)


# ----------------------------------------------------------------------------------------------- reading runs
def run_state(run):
    st = read_json(run / "state.json", {"status": "unknown"})
    if st.get("status") in ("running", "queued") and run.name not in PROCS:
        st["status"] = "interrupted"          # the website was closed while it was running
    return st


def log_info(run):
    p = run / "run.log"
    txt = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    txt = "\n".join(ln.split("\r")[-1] for ln in txt.split("\n"))
    lines = txt.splitlines()
    stage, prog = None, None
    for ln in lines:
        if ln.startswith("[stage] "):
            stage, prog = ln[8:].strip(), None
        m = re.search(r"Processed (\d+)/(\d+) frames", ln) or re.search(r"\| *(\d+)/(\d+) \[", ln)
        if m and int(m.group(2)) > 0:
            prog = {"done": int(m.group(1)), "total": int(m.group(2))}
    return stage, prog, "\n".join(lines[-300:])


def list_files(run):
    out = []
    for top in ("results", "processed_videos", "kinovea"):
        d = run / top
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    out.append({"path": p.relative_to(run).as_posix(), "mb": round(p.stat().st_size / 1e6, 2)})
    for n in sorted(SERVE_FILES):
        if (run / n).is_file():
            out.append({"path": n, "mb": round((run / n).stat().st_size / 1e6, 2)})
    return out


@app.get("/")
def index():
    return send_from_directory(HERE / "static", "index.html")


@app.get("/api/runs")
def list_runs():
    rows = []
    for d in sorted((p for p in DATA.iterdir() if p.is_dir() and not p.name.startswith("_")), key=lambda p: p.stat().st_mtime, reverse=True):
        S = read_json(d / "settings.json", {})
        rows.append({"name": d.name, "created": S.get("created", ""), "status": run_state(d)["status"],
                     "pose_mode": S.get("pose_mode", ""), "has_angles": (d / "angles_all.csv").is_file(),
                     "has_calibration": (d / "results" / "calibration.toml").is_file()})
    return jsonify(rows)


@app.get("/api/runs/<name>")
def run_info(name):
    run = DATA / name
    if not NAME_OK.match(name) or not run.is_dir():
        abort(404)
    stage, prog, log = log_info(run)
    st = run_state(run)
    res = read_json(run / "results" / "results.json", {})
    return jsonify({"name": name, "settings": read_json(run / "settings.json", {}), "status": st["status"], "stage": stage,
                    "progress": prog, "stages": STAGES, "log": log, "files": list_files(run), "folder": str(run),
                    "sync": res.get("sync"), "calib": res.get("calib"), "fps": (res.get("fps") or {}).get("p3d"),
                    "videos": [v for v in ("side_pose_overlay.mp4", "front_pose_overlay.mp4", "side_synced.mp4", "front_synced.mp4")
                               if (run / "processed_videos" / v).is_file()],
                    "overlay_proxy": [v for v in ("side.mp4", "front.mp4") if (run / "results" / v).is_file()],
                    "kinovea": read_json(run / "kinovea" / "sheet.json")})


@app.get("/api/runs/<name>/angles")
def run_angles(name):
    import pandas as pd
    run = DATA / name
    if not NAME_OK.match(name) or not (run / "angles_all.csv").is_file():
        abort(404)
    df = pd.read_csv(run / "angles_all.csv")
    series = {c: [None if v != v else round(float(v), 2) for v in df[c]] for c in df.columns if c not in ("frame", "t")}
    res = read_json(run / "results" / "results.json", {})
    return jsonify({"frame": df["frame"].tolist(), "t": df["t"].tolist(), "series": series,
                    "meta": read_json(run / "angles_meta.json", {}).get("items", []),
                    "fps": (res.get("fps") or {}).get("p3d") or 29.97,
                    "squat_start_side_s": (res.get("sync") or {}).get("squat_start_side_s")})


@app.get("/api/runs/<name>/file/<path:rel>")
def run_file(name, rel):
    run = DATA / name
    if not NAME_OK.match(name) or not run.is_dir():
        abort(404)
    parts = Path(rel).parts
    if not parts or (len(parts) == 1 and rel not in SERVE_FILES) or (len(parts) > 1 and parts[0] not in SERVE_TOP):
        abort(404)
    return send_from_directory(run, rel, conditional=True)     # conditional => video seeking works


@app.post("/api/runs/<name>/kinovea")
def save_kinovea(name):
    """Stores the Kinovea sheet and the comparison result of this run (kinovea/sheet.json, result.txt, pairs.csv)."""
    run = DATA / name
    if not NAME_OK.match(name) or not run.is_dir():
        abort(404)
    body = request.get_json(force=True, silent=True) or {}
    d = run / "kinovea"
    d.mkdir(exist_ok=True)
    write_json(d / "sheet.json", body.get("sheet", {}))
    if body.get("result_text"):
        (d / "result.txt").write_text(str(body["result_text"]), encoding="utf-8")
    if body.get("pairs_csv"):
        (d / "pairs.csv").write_text(str(body["pairs_csv"]), encoding="utf-8")
    return jsonify({"ok": True, "folder": str(d)})


@app.delete("/api/runs/<name>")
def delete_run(name):
    run = DATA / name
    if not NAME_OK.match(name) or not run.is_dir() or run.parent != DATA:
        abort(404)
    p = PROCS.get(name)
    if p and p["proc"].poll() is None:
        return jsonify({"error": "still running"}), 409
    shutil.rmtree(run, ignore_errors=True)       # only this run's own folder; the original videos are never inside it
    return jsonify({"ok": True})


@app.get("/api/calibrations")
def calibrations():
    return jsonify([{"name": d.name, "created": read_json(d / "settings.json", {}).get("created", "")}
                    for d in sorted(DATA.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    if d.is_dir() and not d.name.startswith("_") and (d / "results" / "calibration.toml").is_file()])


if __name__ == "__main__":
    print(f"Pose2Sim Web  ->  http://127.0.0.1:{ARGS.port}     (Ctrl+C to stop)")
    print(f"Runs are stored in: {DATA}")
    app.run(host="127.0.0.1", port=ARGS.port, debug=False, threaded=True)
