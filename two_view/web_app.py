"""
web_app.py -- local website for the two-view squat analysis.

    python two_view/web_app.py        then open  http://127.0.0.1:5000

"Local" means the page runs on YOUR computer only; nothing is uploaded to the internet.

What it does: you upload a front and a side video -> the clap syncs them -> both are analysed with the
existing pipeline -> the page shows both videos together (synced), the angle graphs (click a point to jump
to that moment), and a per-rep table. Past analyses are listed on the left so you can reopen them.
"""
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"
RUNS = OUT / "two_view"          # one folder per analysis
UPLOADS = OUT / "web_uploads"    # raw uploaded videos + run logs
RUNS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3
PROCS = {}  # name -> Popen

def _find_project_python():
    """The analysis needs torch, ultralytics, easydict... which live in the project's venv. Use it automatically,
    even if this website was started with a different Python (e.g. the system one)."""
    for cand in (Path("C:/Proj1Venv/Scripts/python.exe"), ROOT / "Proj1Venv" / "Scripts" / "python.exe",
                 ROOT / "Proj1venv" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    return sys.executable


PROJECT_PY = _find_project_python()
# Pose2Sim lives in its own environment (it needs numpy 2 / another OpenCV, which would break the MotionBERT one).
P2S_PY = next((str(c) for c in (Path("C:/Pose2SimVenv/Scripts/python.exe"),) if c.exists()), PROJECT_PY)
STAGES = ["sync", "front", "side", "fusion", "packaging"]
P2S_STAGES = ["setup", "sync", "calibration", "extract", "pose", "triangulation", "angles", "packaging"]
P2S_CALIB_STAGES = ["setup", "sync", "calibration", "packaging"]
NAME_OK = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
SERVE_OK = {"front.mp4", "side.mp4", "rep_features.csv", "front_angles.csv", "side_angles.csv", "report.html", "results.json",
            "angles.csv", "advancedangles.csv", "reps.csv", "log.txt", "calibration.toml", "points3d.trc",
            "board_side.jpg", "board_front.jpg"}


def slug(text):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", (text or "").strip()).strip("_")[:40]


def _plotly_js():
    try:
        import plotly.offline
        return plotly.offline.get_plotlyjs()
    except Exception:
        pass
    # plotly isn't in this Python -> read the copy that ships inside the project venv (works offline)
    venv_root = Path(PROJECT_PY).parent.parent
    for f in venv_root.glob("Lib/site-packages/plotly/package_data/plotly.min.js"):
        return f.read_text(encoding="utf-8")
    return None


@app.get("/")
def index():
    return send_from_directory(HERE / "web", "index.html")


@app.get("/plotly.js")
def plotly_js():
    js = _plotly_js()
    if js is None:  # plotly not installed here -> let the page fall back to the CDN
        abort(404)
    return Response(js, mimetype="text/javascript", headers={"Cache-Control": "public, max-age=86400"})


def _save_upload(field, dest_dir, stem):
    f = request.files.get(field)
    if f and f.filename:
        dest = dest_dir / f"{stem}{Path(f.filename).suffix.lower() or '.mp4'}"
        f.save(dest)
        return dest
    p = request.form.get(f"{field}_path", "").strip().strip('"').strip("'")
    if p:
        if not Path(p).is_file():
            raise ValueError(f"File not found: {p}")
        return Path(p)
    return None


def _read_meta(run_dir):
    p = run_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _launch(name, cmd, up_dir):
    log = open(up_dir / "log.txt", "w", encoding="utf-8")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    PROCS[name] = {"proc": subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)),
                   "log": log, "t0": time.time()}


def _create_pose2sim_job(name, up_dir, subject, label, kind):
    """Pose2Sim job. kind 'pose2sim' = full 3D analysis, 'p2s_calib' = clap sync + extrinsic calibration only.
    Videos: side + front (board -> clap [-> squats]); intrinsic videos unless an earlier calibration is reused."""
    f = request.form
    for key, what in (("intr_corners", "Intrinsic board corners"), ("ext_corners", "Extrinsic board corners")):
        nums = re.findall(r"\d+", f.get(key, "6,7"))
        if len(nums) != 2 or min(map(int, nums)) < 2:
            return jsonify({"error": f"{what}: give two numbers of inner corners, e.g. 6,7."}), 400
    for key, what in (("intr_square", "Intrinsic square size"), ("ext_square", "Extrinsic square size")):
        try:
            if float(f.get(key, "60")) <= 0:
                raise ValueError
        except ValueError:
            return jsonify({"error": f"{what}: give the square size in millimetres, e.g. 60."}), 400
    src = f.get("calib_source", "new")
    reuse = None
    if src != "new":
        reuse = RUNS / src / "calibration.toml"
        if not NAME_OK.match(src) or not reuse.is_file():
            return jsonify({"error": f"No calibration found for '{src}'."}), 400
    try:
        fields = ("side", "front") + (("intr_side", "intr_front") if reuse is None else ())
        vids = {k: _save_upload(k, up_dir, k) for k in fields}
        if any(v is None for v in vids.values()):
            raise ValueError("Please choose the side and front videos" +
                             (" and both intrinsic videos." if reuse is None else "."))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    run_dir = RUNS / name
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(
        {"name": name, "kind": kind, "subject": subject, "label": label, "calib_source": src,
         **{f"{k}_file": v.name for k, v in vids.items()}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}))
    cmd = [P2S_PY, "-u", str(HERE / "run_pose2sim.py"), "--name", name,
           "--mode", "extrinsic" if kind == "p2s_calib" else "full",
           "--side", str(vids["side"]), "--front", str(vids["front"]),
           "--intr-corners", f.get("intr_corners", "6,7"), "--intr-square", f.get("intr_square", "60"),
           "--ext-corners", f.get("ext_corners", "6,7"), "--ext-square", f.get("ext_square", "60"),
           "--clap-window", f.get("clap_window", "90"),
           "--squat-claps", "2" if f.get("squat_claps") == "2" and kind == "pose2sim" else "1",
           "--board-position", "horizontal" if f.get("board_position") == "horizontal" else "vertical"]
    if reuse is None:
        cmd += ["--intr-side", str(vids["intr_side"]), "--intr-front", str(vids["intr_front"])]
    else:  # full analysis: whole calibration; calibration-only run: just its intrinsics
        cmd += ["--calib-from" if kind == "pose2sim" else "--intr-from", str(reuse)]
    _launch(name, cmd, up_dir)
    return jsonify({"name": name})


@app.get("/api/calibrations")
def list_calibrations():
    """Earlier runs whose calibration can be reused (they all contain intrinsics + extrinsics)."""
    rows = []
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "calibration.toml").is_file() and (d / "results.json").is_file():
            meta = _read_meta(d)
            rows.append({"name": d.name, "kind": meta.get("kind", ""), "created": meta.get("created", "")})
    return jsonify(rows)


@app.post("/api/jobs")
def create_job():
    label, subject = request.form.get("label", "").strip(), request.form.get("subject", "").strip()
    base = slug(request.form.get("name")) or slug(f"{subject}_{label}") or "run"
    name = f"{base}_{time.strftime('%m%d-%H%M%S')}"
    n = 2
    while (UPLOADS / name).exists() or (RUNS / name).exists():  # two jobs started in the same second
        name = f"{base}_{time.strftime('%m%d-%H%M%S')}_{n}"
        n += 1
    up_dir = UPLOADS / name
    up_dir.mkdir(parents=True)
    if request.form.get("method") in ("pose2sim", "p2s_calib"):
        resp = _create_pose2sim_job(name, up_dir, subject, label, request.form["method"])
        if isinstance(resp, tuple):  # validation error: don't leave an empty upload folder behind
            import shutil
            shutil.rmtree(up_dir, ignore_errors=True)
        return resp
    # The "Classic" mode (single-view AlphaPose pipeline on each video, fused per rep: run_two_view.py) was removed
    # with AlphaPose on 2026-10-18; its earlier runs can still be opened.
    import shutil
    shutil.rmtree(up_dir, ignore_errors=True)
    return jsonify({"error": "The Classic mode was removed. Use 3D analysis (Pose2Sim)."}), 400


def _status(name):
    run_dir = RUNS / name
    if not run_dir.is_dir():
        abort(404)
    done = (run_dir / "results.json").exists()
    info = PROCS.get(name)
    meta = _read_meta(run_dir)
    log_text = ""
    if info:
        info["log"].flush()
    for lp in (UPLOADS / name / "log.txt", run_dir / "log.txt"):
        if lp.exists():
            log_text = lp.read_text(encoding="utf-8", errors="replace")
            break
    code = info["proc"].poll() if info else (0 if done else -1)  # -1: server restarted mid-run
    stage, prog = None, None
    for line in log_text.splitlines():
        if line.startswith("[stage] "):
            stage, prog = line[8:].strip(), None
        m = re.search(r"Processed (\d+)/(\d+) frames", line) or re.search(r"\| *(\d+)/(\d+) \[", line)  # pipeline / tqdm
        if m and int(m.group(2)) > 0:
            prog = {"done": int(m.group(1)), "total": int(m.group(2))}
    state = "done" if done else ("running" if code is None else "error")
    stages = {"pose2sim": P2S_STAGES, "p2s_calib": P2S_CALIB_STAGES}.get(meta.get("kind"), STAGES)
    return {"name": name, "state": state, "stage": stage, "progress": prog, "stages": stages,
            "kind": meta.get("kind", "classic"),
            "elapsed_s": round(time.time() - info["t0"]) if info else None,
            "log": "\n".join(log_text.splitlines()[-250:])}


@app.get("/api/jobs/<name>")
def job_status(name):
    if not NAME_OK.match(name):
        abort(404)
    return jsonify(_status(name))


@app.get("/api/jobs")
def list_jobs():
    rows = []
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = _read_meta(d)
        rows.append({"name": d.name, "done": (d / "results.json").exists(), "kind": meta.get("kind", "classic"),
                     "subject": meta.get("subject", ""), "label": meta.get("label", ""),
                     "created": meta.get("created", time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)))})
    return jsonify(rows[:40])


@app.get("/api/jobs/<name>/file/<fname>")
def job_file(name, fname):
    if not NAME_OK.match(name) or fname not in SERVE_OK:
        abort(404)
    return send_from_directory(RUNS / name, fname, conditional=True)  # conditional => video seeking works


@app.delete("/api/jobs/<name>")
def delete_job(name):
    import shutil
    if not NAME_OK.match(name) or not (RUNS / name).is_dir():
        abort(404)
    p = PROCS.get(name)
    if p and p["proc"].poll() is None:
        return jsonify({"error": "still running"}), 409
    if p:
        p["log"].close()  # Windows keeps an open log.txt locked, which would leave the upload folder behind
        PROCS.pop(name, None)
    shutil.rmtree(RUNS / name, ignore_errors=True)
    shutil.rmtree(UPLOADS / name, ignore_errors=True)
    for view in ("front", "side"):
        shutil.rmtree(OUT / f"{name}_{view}", ignore_errors=True)
    shutil.rmtree(OUT / "pose2sim" / name, ignore_errors=True)  # Pose2Sim project folder
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"Analysis will run with: {PROJECT_PY}")
    print(f"Pose2Sim analysis will run with: {P2S_PY}")
    if PROJECT_PY == sys.executable:
        print("WARNING: project environment C:\\Proj1Venv not found; using the current Python. "
              "Run two_view\\start_website.ps1 to create it.")
    print("Open http://127.0.0.1:5000 in your browser.  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
