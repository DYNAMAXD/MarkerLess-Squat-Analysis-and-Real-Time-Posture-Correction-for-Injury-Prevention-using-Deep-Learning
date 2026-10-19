"""
product_app.py -- the final product: upload a front + a side squat video, get the errors per rep, the 3D body with
the error's saliency on it, the keypoint videos and the rep table, all on one timeline.

    .venv\\app\\Scripts\\python.exe final_product/product_app.py        then open http://127.0.0.1:5090
    (or run final_product\\final_product.ps1, which also checks the environments)

Pages (one HTML page, web/product.html + web/product.js):
  New analysis   pick a camera profile, give the two videos -> a session; jobs run one at a time in the background
  Session        processing progress, then the results (mesh + saliency, videos, errors and fixes, timeline,
                 compare two reps, download)
  History        all sessions with their score and how the errors / depth / tempo change over time
  Cameras        camera profiles: intrinsics made once from the user's intrinsic videos (or imported), plus an
                 optional fallback calibration from extrinsic videos

Everything lives in outputs/product/ (sessions/, cameras/, _work/). The work is done by product_pipeline.py.
The error model is final_product/squat_model_export/ (in git); without it the sessions show the reps without errors
and "Try demo predictions" fills in made-up errors (clearly marked DEMO). Camera calibrations shipped in
final_product/presets/cameras/ are copied into outputs/product/cameras/ on the first start.
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import product_pipeline as PL  # noqa: E402
import product_predictions as PP  # noqa: E402

PRODUCT = PL.PRODUCT
SESSIONS = PRODUCT / "sessions"
CAMERAS = PRODUCT / "cameras"
for _d in (SESSIONS, CAMERAS):
    _d.mkdir(parents=True, exist_ok=True)
# painted saliency templates (mesh_saliency_painter.html): the env folder, else both usual places
TEMPLATE_DIRS = [Path(os.environ["SALIENCY_TEMPLATES"])] if os.environ.get("SALIENCY_TEMPLATES") else \
    [ROOT / "two_view" / "saliency_templates", ROOT / "saliency_maps"]
SMPL_PKL = ROOT / "MotionBERT" / "data" / "mesh" / "SMPL_NEUTRAL.pkl"
CONTENT = HERE / "product_content.json"
ID_OK = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
FILES_OK = {"front.mp4", "side.mp4", "mesh/verts.bin", "predictions.json", "results.json", "rep_results.csv",
            "angles.csv", "advancedangles.csv", "points3d.trc", "calibration.toml", "log.txt", "pipeline_log.txt",
            "board_front.jpg", "board_side.jpg"}
ZIP_FILES = ["rep_results.csv", "predictions.json", "angles.csv", "advancedangles.csv", "points3d.trc", "front.mp4",
             "side.mp4", "calibration.toml", "log.txt"]
MAX_UPLOAD = 8 * 1024 ** 3
# PRODUCT_ALLOW_PATHS=0 (hosted, e.g. the Hugging Face Space): uploads only, no paths of files on the server
ALLOW_PATHS = os.environ.get("PRODUCT_ALLOW_PATHS", "1") != "0"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
PROCS = {}                 # (kind, id) -> Popen
LOCK = threading.Lock()
_faces = {"bin": None}


def slug(text, n=40):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", (text or "").strip()).strip("_")[:n]


def job_dir(kind, jid):
    if not ID_OK.match(jid or ""):
        abort(404)
    d = (SESSIONS if kind == "session" else CAMERAS) / jid
    if not d.is_dir():
        abort(404)
    return d


def new_id(base, parent):
    jid = f"{slug(base) or 'run'}_{time.strftime('%Y%m%d-%H%M%S')}"
    n = 2
    while (parent / jid).exists():
        jid = f"{slug(base) or 'run'}_{time.strftime('%Y%m%d-%H%M%S')}_{n}"
        n += 1
    return jid


# --------------------------------------------------------------------------- job queue (one at a time)
def _pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def job_status(kind, d):
    """status.json, with a dead 'running' job turned into 'interrupted'."""
    st = PL.read_json(d / "status.json", {}) or {}
    if st.get("state") == "running":
        p = PROCS.get((kind, d.name))
        alive = p.poll() is None if p is not None else _pid_alive(st.get("pid"))
        if not alive:
            st["state"] = "interrupted"
            st["error"] = st.get("error") or "The job stopped unexpectedly (window closed or computer restarted). Press Retry."
    return st


def _all_jobs():
    for kind, parent in (("camera", CAMERAS), ("session", SESSIONS)):
        for d in parent.iterdir():
            if d.is_dir():
                yield kind, d


def _session_camera(d):
    return (PL.read_json(d / "session.json", {}) or {}).get("camera")


def _queue_loop():
    while True:
        try:
            with LOCK:
                jobs = [(k, d, job_status(k, d)) for k, d in _all_jobs()]
                if not any(st.get("state") == "running" for _, _, st in jobs):
                    busy_cams = {d.name for k, d, st in jobs if k == "camera" and st.get("state") in ("queued", "running")}
                    queued = sorted((st.get("queued_at", 0), k, d) for k, d, st in jobs if st.get("state") == "queued"
                                    and not (k == "session" and _session_camera(d) in busy_cams))  # calibration first
                    if queued:
                        _, kind, d = queued[0]
                        PROCS[(kind, d.name)] = subprocess.Popen(
                            [sys.executable, "-u", str(HERE / "product_pipeline.py"), kind, str(d)],
                            stdout=open(d / "job_console.txt", "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                            cwd=str(ROOT))
                        for _ in range(50):  # wait until the job has taken over status.json
                            if (PL.read_json(d / "status.json", {}) or {}).get("state") != "queued":
                                break
                            time.sleep(0.1)
        except Exception as e:  # noqa: BLE001
            print("queue:", e, flush=True)
        time.sleep(2)


def enqueue(d, stages):
    old = PL.read_json(d / "status.json", {}) or {}
    PL.write_json(d / "status.json", {"state": "queued", "queued_at": time.time(), "stages": stages,
                                      "done": [], "warnings": old.get("warnings", []), "stage": None})


def queue_position(kind, d):
    with LOCK:
        q = sorted((st.get("queued_at", 0), dd.name) for k, dd, st in
                   ((k, dd, PL.read_json(dd / "status.json", {}) or {}) for k, dd in _all_jobs()) if st.get("state") == "queued")
    names = [n for _, n in q]
    return names.index(d.name) + 1 if d.name in names else None


# --------------------------------------------------------------------------- uploads
def save_video(field, dest_dir, stem):
    """An uploaded file, or a path typed in the '<field>_path' box (videos already on this PC: no copy)."""
    f = request.files.get(field)
    if f and f.filename:
        dest = dest_dir / f"{stem}{Path(f.filename).suffix.lower() or '.mp4'}"
        f.save(dest)
        return str(dest)
    p = request.form.get(f"{field}_path", "").strip().strip('"').strip("'") if ALLOW_PATHS else ""
    if p:
        if not Path(p).is_file():
            raise ValueError(f"File not found: {p}")
        return str(Path(p).resolve())
    return None


# --------------------------------------------------------------------------- pages / static
@app.get("/")
def index():
    return send_from_directory(HERE / "web", "product.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/product.js")
def product_js():
    return send_from_directory(HERE / "web", "product.js")


@app.get("/static/<name>")
def static_file(name):
    if name not in ("product.js",):
        abort(404)
    return send_from_directory(HERE / "web", name)


@app.get("/vendor/<path:name>")
def vendor(name):
    return send_from_directory(HERE / "web" / "vendor", name)


@app.get("/api/content")
def content():
    return send_file(CONTENT, mimetype="application/json", max_age=0)


@app.get("/api/info")
def info():
    return jsonify({"model_active": (PL.MODEL_DIR / "model.pt").exists() and PL.MODEL_PY.exists(),
                    "model_dir": str(PL.MODEL_DIR), "smpl": SMPL_PKL.exists(), "pose2sim": bool(PL.P2S_PY),
                    "pose_mode": PL.POSE_MODE, "templates": [str(f) for f in TEMPLATE_DIRS], "allow_paths": ALLOW_PATHS})


@app.get("/api/faces.bin")
def faces():
    """SMPL triangles as little-endian uint16 (13776 x 3)."""
    if _faces["bin"] is None:
        if not SMPL_PKL.exists():
            abort(404)
        import numpy as np
        import smplx
        f = smplx.create(str(SMPL_PKL), model_type="smpl", gender="neutral").faces
        _faces["bin"] = np.asarray(f, dtype="<u2").tobytes()
    return Response(_faces["bin"], mimetype="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=86400"})


def _tkey(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def template_files():
    """error key (lower-case letters/digits of the template's mesh_name) -> file; the first folder wins."""
    out = {}
    for folder in reversed(TEMPLATE_DIRS):
        for t in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                d = json.loads(t.read_text(encoding="utf-8"))
                if len(d.get("saliency") or []) == 6890:
                    out[_tkey(d.get("mesh_name") or re.sub(r"_saliency$", "", t.stem))] = t
            except (OSError, ValueError):
                pass
    return out


@app.get("/api/templates")
def templates():
    return jsonify({"folders": [str(f) for f in TEMPLATE_DIRS], "templates": {k: k for k in template_files()}})


@app.get("/api/templates/<key>")
def template(key):
    p = template_files().get(_tkey(key))
    if p is None:
        abort(404)
    return send_file(p, mimetype="application/json")


# --------------------------------------------------------------------------- camera profiles
def _intrinsics_summary(toml_path):
    import tomllib
    try:
        d = tomllib.loads(Path(toml_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = []
    for k, v in d.items():
        if not isinstance(v, dict) or "matrix" not in v:
            continue
        (fx, _, cx), (_, fy, cy) = v["matrix"][0], v["matrix"][1]
        w, h = v.get("size", [0, 0])
        warn = []
        if abs(fx - fy) / max(fx, 1) > 0.02:
            warn.append("fx and fy differ by more than 2 %")
        if w and (abs(cx - w / 2) > 0.1 * w or abs(cy - h / 2) > 0.1 * h):
            warn.append("optical centre far from the image centre")
        if w and fx < 0.25 * w:
            warn.append("focal length implausibly small")
        rows.append({"camera": "front" if "front" in k else "side" if "side" in k else k, "size": [w, h],
                     "fx": round(fx, 1), "fy": round(fy, 1), "cx": round(cx, 1), "cy": round(cy, 1), "warnings": warn})
    return rows


def _camera_row(d):
    prof = PL.read_json(d / "profile.json", {}) or {}
    st = job_status("camera", d)
    ready = (d / "intrinsics.toml").exists()
    return {"id": d.name, "name": prof.get("name", d.name), "created": prof.get("created", ""),
            "source": prof.get("source", ""), "board": prof.get("board", {}),
            "state": "ready" if ready and st.get("state") in (None, "done") else st.get("state") or ("ready" if ready else "error"),
            "ready": ready, "fallback": (d / "fallback.toml").exists(),
            "intrinsics": _intrinsics_summary(d / "intrinsics.toml") if ready else [],
            "status": st, "queue": queue_position("camera", d) if st.get("state") == "queued" else None}


def _board_from_form(f):
    board = {}
    for key, what in (("intr_corners", "Intrinsic board corners"), ("ext_corners", "Extrinsic board corners")):
        nums = re.findall(r"\d+", f.get(key, "6,7"))
        if len(nums) != 2 or min(map(int, nums)) < 2:
            raise ValueError(f"{what}: give two numbers of inner corners, e.g. 6,7.")
        board[key] = f"{nums[0]},{nums[1]}"
    for key, what in (("intr_square", "Intrinsic square size"), ("ext_square", "Extrinsic square size")):
        try:
            v = float(f.get(key, "41"))
            if v <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"{what}: give the square size in millimetres, e.g. 41.") from None
        board[key] = v
    board["board_position"] = "horizontal" if f.get("board_position") == "horizontal" else "vertical"
    return board


@app.get("/api/cameras")
def cameras():
    rows = [_camera_row(d) for d in sorted(CAMERAS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir()]
    return jsonify(rows)


def make_camera(f, name, source):
    """A camera profile from the request's form: source 'import' (calibration .toml, ready at once) or 'videos'
    (intrinsic videos, calibrated by a queued job). Returns its id; raises ValueError (nothing left behind)."""
    board = _board_from_form(f)
    jid = new_id(name, CAMERAS)
    d = CAMERAS / jid
    d.mkdir(parents=True)
    up = d / "uploads"
    up.mkdir()
    try:
        prof = {"id": jid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "board": board}
        if source == "import":
            toml = request.files.get("calib")
            if not toml or not toml.filename:
                raise ValueError("Choose a calibration.toml file to import.")
            text = toml.read().decode("utf-8", errors="replace")
            (d / "intrinsics.toml").write_text(text, encoding="utf-8")
            if not _intrinsics_summary(d / "intrinsics.toml"):
                raise ValueError("That file has no camera intrinsics (expected a Pose2Sim calibration .toml).")
            fb = request.files.get("fallback")
            if fb and fb.filename:
                (d / "fallback.toml").write_bytes(fb.read())
            prof.update(source="import")
            PL.write_json(d / "profile.json", prof)
            PL.write_json(d / "status.json", {"state": "done", "stages": PL.CAMERA_STAGES, "done": PL.CAMERA_STAGES,
                                             "warnings": [], "finished": time.time()})
            return jid
        vids = {k: save_video(k, up, k) for k in ("intr_front", "intr_side", "ext_front", "ext_side")}
        if not (vids["intr_front"] and vids["intr_side"]):
            raise ValueError("Give both intrinsic videos (front and side camera).")
        if bool(vids["ext_front"]) != bool(vids["ext_side"]):
            raise ValueError("Fallback calibration needs both extrinsic videos (or neither).")
        prof.update(source="videos", videos=vids)
        PL.write_json(d / "profile.json", prof)
        enqueue(d, PL.CAMERA_STAGES)
        return jid
    except ValueError:
        shutil.rmtree(d, ignore_errors=True)
        raise


def camera_usable(cam):
    """Calibrated, or being calibrated (a session queued on it waits for the calibration)."""
    d = CAMERAS / cam
    if not ID_OK.match(cam or "") or not d.is_dir():
        return False
    return (d / "intrinsics.toml").exists() or job_status("camera", d).get("state") in ("queued", "running")


@app.post("/api/cameras")
def create_camera():
    f = request.form
    try:
        jid = make_camera(f, f.get("name", "").strip() or "My cameras", f.get("source"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"id": jid})


@app.get("/api/cameras/<jid>")
def camera(jid):
    d = job_dir("camera", jid)
    row = _camera_row(d)
    row["log"] = PL.read_log(d / "pipeline_log.txt").splitlines()[-40:]
    return jsonify(row)


@app.post("/api/cameras/<jid>/retry")
def camera_retry(jid):
    d = job_dir("camera", jid)
    if job_status("camera", d).get("state") in ("running", "queued"):
        return jsonify({"error": "already running"}), 409
    enqueue(d, PL.CAMERA_STAGES)
    return jsonify({"ok": True})


@app.delete("/api/cameras/<jid>")
def delete_camera(jid):
    d = job_dir("camera", jid)
    if job_status("camera", d).get("state") == "running":
        return jsonify({"error": "This profile is still being calibrated. Stop it first."}), 409
    shutil.rmtree(d, ignore_errors=True)
    for w in PL.WORK.glob(f"cam_{jid}_*"):
        shutil.rmtree(w, ignore_errors=True)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- sessions
def _session_row(d):
    sess = PL.read_json(d / "session.json", {}) or {}
    st = job_status("session", d)
    return {"id": d.name, "name": sess.get("name", d.name), "created": sess.get("created", ""),
            "camera": sess.get("camera"), "state": st.get("state", "unknown"), "stage": st.get("stage"),
            "summary": sess.get("summary"), "error": st.get("error"),
            "queue": queue_position("session", d) if st.get("state") == "queued" else None}


@app.get("/api/sessions")
def sessions():
    rows = [_session_row(d) for d in sorted(SESSIONS.iterdir(), key=lambda p: p.name) if d.is_dir()]
    rows.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(rows)


@app.post("/api/sessions")
def create_session():
    """cam_mode 'saved' (camera = a profile's id) or 'videos' / 'import': the calibration comes with the session
    (intrinsic videos or a .toml) and becomes a new camera profile, calibrated before the session runs."""
    f = request.form
    mode = f.get("cam_mode", "saved")
    cam = f.get("camera", "")
    if mode == "saved" and not camera_usable(cam):
        return jsonify({"error": "Choose a camera calibration (saved, new intrinsic videos or a .toml file)."}), 400
    name = f.get("name", "").strip() or time.strftime("Session %d %b %H:%M")
    jid = new_id(name, SESSIONS)
    d = SESSIONS / jid
    up = d / "uploads"
    up.mkdir(parents=True)
    try:
        front, side = save_video("front", up, "front_in"), save_video("side", up, "side_in")
        if not front or not side:
            raise ValueError("Give both videos: front and side.")
        if Path(front).resolve() == Path(side).resolve():
            raise ValueError("The front and side video are the same file.")
        if mode in ("videos", "import"):
            cam = make_camera(f, f.get("camera_name", "").strip() or time.strftime("Cameras %d %b %H:%M"), mode)
    except ValueError as e:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify({"error": str(e)}), 400
    PL.write_json(d / "session.json", {"id": jid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "camera": cam, "camera_new": mode in ("videos", "import"),
                                       "front": front, "side": side})
    enqueue(d, PL.SESSION_STAGES)
    return jsonify({"id": jid})


@app.get("/api/sessions/<jid>")
def session(jid):
    d = job_dir("session", jid)
    row = _session_row(d)
    st = job_status("session", d)
    row["status"] = st
    row["log"] = PL.read_log(d / "pipeline_log.txt").splitlines()[-60:]
    if st.get("state") == "running" and st.get("stage") in PL.SESSION_STAGES[:7]:
        cons = sorted(d.glob("preprocess_console_*.txt"))
        if cons:
            row["log"] = PL.read_log(cons[-1]).splitlines()[-60:]
    row["elapsed_s"] = round(((st.get("finished") or time.time()) - st["started"])) if st.get("started") else None
    cd = CAMERAS / (row.get("camera") or "")
    if row.get("camera") and ID_OK.match(row["camera"]) and cd.is_dir():
        cst = job_status("camera", cd)
        row["camera_job"] = {"name": (PL.read_json(cd / "profile.json", {}) or {}).get("name", cd.name),
                             "ready": (cd / "intrinsics.toml").exists(), "state": cst.get("state"),
                             "stage": cst.get("stage"), "progress": cst.get("progress"), "error": cst.get("error")}
    return jsonify(row)


@app.get("/api/sessions/<jid>/data")
def session_data(jid):
    """Everything the results page needs (except the large files)."""
    d = job_dir("session", jid)
    return jsonify(PP.session_view(d, SMPL_PKL.exists(), job_status("session", d).get("warnings", [])))


@app.get("/api/sessions/<jid>/file/<path:name>")
def session_file(jid, name):
    d = job_dir("session", jid)
    if name not in FILES_OK or not (d / name).is_file():
        abort(404)
    return send_file(d / name, conditional=True,
                     mimetype="application/octet-stream" if name.endswith(".bin") else None)


@app.get("/api/sessions/<jid>/reps.csv")
def session_reps_csv(jid):
    d = job_dir("session", jid)
    pred = PL.read_json(d / "predictions.json")
    if not pred:
        abort(404)
    return Response(PP.user_reps_csv(pred), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{jid}_reps.csv"'})


@app.get("/api/sessions/<jid>/download")
def session_zip(jid):
    d = job_dir("session", jid)
    pred = PL.read_json(d / "predictions.json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if pred:
            z.writestr("reps.csv", PP.user_reps_csv(pred))
        for name in ZIP_FILES:
            if (d / name).is_file():
                z.write(d / name, name, compress_type=zipfile.ZIP_STORED if name.endswith(".mp4") else zipfile.ZIP_DEFLATED)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{jid}.zip")


@app.post("/api/sessions/<jid>/demo")
def session_demo(jid):
    """Made-up predictions to try the page while the model is not trained (on: true) / back to none (on: false)."""
    d = job_dir("session", jid)
    if not (d / "results.json").exists():
        return jsonify({"error": "The session is not finished."}), 409
    cur = PL.read_json(d / "predictions.json", {}) or {}
    if cur.get("model_active"):
        return jsonify({"error": "This session has real model predictions."}), 409
    on = bool((request.get_json(silent=True) or {}).get("on"))
    pred = PP.demo_predictions(d, seed=int(time.time()) % 1000) if on else PP.inactive_predictions(d)
    sess = PL.read_json(d / "session.json", {}) or {}
    sess["summary"] = PP.write_all(d, pred)
    PL.write_json(d / "session.json", sess)
    return jsonify({"ok": True})


@app.post("/api/sessions/<jid>/retry")
def session_retry(jid):
    d = job_dir("session", jid)
    if job_status("session", d).get("state") in ("running", "queued"):
        return jsonify({"error": "already running"}), 409
    enqueue(d, PL.SESSION_STAGES)
    return jsonify({"ok": True})


@app.post("/api/<kind>/<jid>/cancel")
def cancel(kind, jid):
    kind = {"sessions": "session", "cameras": "camera"}.get(kind)
    if not kind:
        abort(404)
    d = job_dir(kind, jid)
    _cancel(kind, d)
    sess = (PL.read_json(d / "session.json", {}) or {}) if kind == "session" else {}
    cd = CAMERAS / (sess.get("camera") or "")
    if sess.get("camera_new") and ID_OK.match(sess.get("camera") or "") and cd.is_dir():
        _cancel("camera", cd)   # the calibration started with this session stops with it
    return jsonify({"ok": True})


def _cancel(kind, d):
    st = job_status(kind, d)
    if st.get("state") == "queued":
        st.update(state="cancelled", error="Stopped before it started.")
        PL.write_json(d / "status.json", st)
    elif st.get("state") == "running":
        (d / "CANCEL").write_text("stop", encoding="utf-8")


@app.delete("/api/sessions/<jid>")
def delete_session(jid):
    d = job_dir("session", jid)
    if job_status("session", d).get("state") == "running":
        return jsonify({"error": "This session is still running. Stop it first."}), 409
    p = PROCS.pop(("session", jid), None)
    if p and p.stdout:
        p.stdout.close()
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(PL.WORK / jid, ignore_errors=True)
    return jsonify({"ok": True})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Upload too large (max {MAX_UPLOAD // 1024 ** 3} GB)." + (" Type the file path instead." if ALLOW_PATHS else "")}), 413


def install_presets():
    """Camera calibrations shipped with the app (PRODUCT_PRESETS, default final_product/presets/cameras/ = a folder of
    profile folders, each with profile.json, status.json, intrinsics.toml): copied into the data folder once, so a new
    user only needs the squat videos when they record with the dataset's cameras."""
    src = os.environ.get("PRODUCT_PRESETS") or str(HERE / "presets" / "cameras")
    for p in sorted(Path(src).iterdir()) if src and Path(src).is_dir() else []:
        if p.is_dir() and (p / "intrinsics.toml").exists() and not (CAMERAS / p.name).exists():
            shutil.copytree(p, CAMERAS / p.name)
            print(f"  preset calibration: {p.name}")


if __name__ == "__main__":
    port = int(os.environ.get("PRODUCT_PORT", "5090"))
    host = os.environ.get("PRODUCT_HOST", "127.0.0.1")    # 0.0.0.0 in the Docker image
    install_presets()
    import socket
    with socket.socket() as s:  # on Windows a second server can bind the same port and old code keeps answering
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"The app is already running at http://127.0.0.1:{port} - close that window first (or just open the address).")
            sys.exit(1)
    threading.Thread(target=_queue_loop, daemon=True).start()
    print(f"Squat analysis: http://127.0.0.1:{port}   (keep this window open; closing it stops running analyses)")
    print(f"  data: {PRODUCT}\n  saliency templates: {', '.join(map(str, TEMPLATE_DIRS))}\n  model: {PL.MODEL_DIR} "
          f"({'found' if (PL.MODEL_DIR / 'model.pt').exists() else 'not trained yet - errors are not detected'})")
    app.run(host=host, port=port, debug=False, threaded=True)
