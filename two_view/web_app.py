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
STAGES = ["sync", "front", "side", "fusion", "packaging"]
NAME_OK = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
SERVE_OK = {"front.mp4", "side.mp4", "rep_features.csv", "front_angles.csv", "side_angles.csv", "report.html", "results.json"}


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


@app.post("/api/jobs")
def create_job():
    label, subject = request.form.get("label", "").strip(), request.form.get("subject", "").strip()
    base = slug(request.form.get("name")) or slug(f"{subject}_{label}") or "run"
    name = f"{base}_{time.strftime('%m%d-%H%M%S')}"
    up_dir = UPLOADS / name
    up_dir.mkdir(parents=True)
    try:
        front, side = _save_upload("front", up_dir, "front"), _save_upload("side", up_dir, "side")
        if front is None or side is None:
            raise ValueError("Please choose both a front and a side video.")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    run_dir = RUNS / name
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(
        {"name": name, "subject": subject, "label": label, "front_file": front.name, "side_file": side.name,
         "created": time.strftime("%Y-%m-%d %H:%M:%S")}))
    cmd = [PROJECT_PY, "-u", str(HERE / "run_two_view.py"), "--name", name,
           "--front", str(front), "--side", str(side), "--device", request.form.get("device", "auto")]
    if request.form.get("sync_only") == "1":
        cmd.append("--sync-only")
    log = open(up_dir / "log.txt", "w", encoding="utf-8")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    PROCS[name] = {"proc": subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)),
                   "log": log, "t0": time.time()}
    return jsonify({"name": name})


def _status(name):
    run_dir = RUNS / name
    if not run_dir.is_dir():
        abort(404)
    done = (run_dir / "results.json").exists()
    info = PROCS.get(name)
    log_text = ""
    if info:
        info["log"].flush()
        log_text = (UPLOADS / name / "log.txt").read_text(encoding="utf-8", errors="replace")
    code = info["proc"].poll() if info else (0 if done else -1)  # -1: server restarted mid-run
    stage, prog = None, None
    for line in log_text.splitlines():
        if line.startswith("[stage] "):
            stage, prog = line[8:].strip(), None
        m = re.search(r"Processed (\d+)/(\d+) frames", line)
        if m:
            prog = {"done": int(m.group(1)), "total": int(m.group(2))}
    state = "done" if done else ("running" if code is None else "error")
    return {"name": name, "state": state, "stage": stage, "progress": prog, "stages": STAGES,
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
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        rows.append({"name": d.name, "done": (d / "results.json").exists(),
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
    shutil.rmtree(RUNS / name, ignore_errors=True)
    shutil.rmtree(UPLOADS / name, ignore_errors=True)
    for view in ("front", "side"):
        shutil.rmtree(OUT / f"{name}_{view}", ignore_errors=True)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"Analysis will run with: {PROJECT_PY}")
    if PROJECT_PY == sys.executable:
        print("WARNING: project environment C:\\Proj1Venv not found; using the current Python. "
              "Run two_view\\start_website.ps1 to create it.")
    print("Open http://127.0.0.1:5000 in your browser.  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
