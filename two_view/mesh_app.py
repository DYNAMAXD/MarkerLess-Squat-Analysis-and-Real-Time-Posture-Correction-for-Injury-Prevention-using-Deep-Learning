"""
mesh_app.py -- local web page: the SMPL body mesh fitted to the Pose2Sim 3D points (like MotionBERT's mesh output),
played back in 3D next to the front / side videos, with each labelled error's painted saliency template overlaid on
the moving body.

    C:\\Proj1Venv\\Scripts\\python.exe two_view/mesh_app.py        then open http://127.0.0.1:5070
    (or run two_view\\start_mesh_viewer.ps1)

Folders it can open: website runs (outputs/two_view/<run>/) and batch outputs (<dataset>/preprocessed_data/<person>/),
anything holding points3d.trc. "Build 3D mesh" runs p2s_mesh.py on that folder (~1 min on a CPU).

Saliency templates: <fault>_saliency.json files painted in mesh_saliency_painter.html (6890 values on the SMPL
T-pose, same vertex order as the fitted mesh), kept in two_view/saliency_templates/ (upload them on the page).
A template is shown for a person when its name matches one of the person's errors (meta.json "errors", from the
CSV error_performed; case, spaces and punctuation are ignored).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATES = Path(os.environ.get("SALIENCY_TEMPLATES", HERE / "saliency_templates"))
TEMPLATES.mkdir(parents=True, exist_ok=True)
SMPL_PKL = ROOT / "MotionBERT" / "data" / "mesh" / "SMPL_NEUTRAL.pkl"
FILES_OK = {"front.mp4", "side.mp4", "mesh/verts.bin", "board_side.jpg", "board_front.jpg", "mesh/log.txt"}
app = Flask(__name__)
BUILDS = {}           # folder -> Popen
_faces = {"bin": None}


def key(name):
    """Error / template name -> matching key: lower case letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def template_key(path):
    try:
        name = json.loads(Path(path).read_text(encoding="utf-8")).get("mesh_name") or ""
    except (OSError, ValueError):
        name = ""
    return key(name or re.sub(r"_saliency$", "", Path(path).stem))


def folder_arg():
    p = (request.args.get("path") or (request.get_json(silent=True) or {}).get("path") or "").strip().strip('"')
    if not p or not (Path(p) / "points3d.trc").is_file():
        abort(404)
    return Path(p).resolve()


def person_errors(folder):
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], {}
    errs = meta.get("errors")
    if not errs:  # website runs: the "label" field, possibly comma separated
        errs = [e.strip() for e in str(meta.get("error_performed") or meta.get("label") or "").split(",") if e.strip()]
    return errs, meta


def mesh_state(folder):
    p = BUILDS.get(str(folder))
    if p is not None and p.poll() is None:
        return "building"
    if (folder / "mesh" / "done.txt").exists():
        return "done"
    if (folder / "mesh" / "log.txt").exists():
        return "error"
    return "none"


@app.get("/")
def index():
    return send_from_directory(HERE / "web", "mesh.html")


@app.get("/vendor/<path:name>")
def vendor(name):
    return send_from_directory(HERE / "web" / "vendor", name)


@app.get("/painter")
def painter():
    return send_from_directory(ROOT, "mesh_saliency_painter.html")


@app.get("/smpl_neutral_tpose.obj")
def tpose_obj():
    return send_from_directory(ROOT, "smpl_neutral_tpose.obj", as_attachment=True)


@app.get("/api/folders")
def folders():
    """Run / person folders (holding points3d.trc) directly inside `root` (or `root` itself)."""
    root = Path((request.args.get("root") or str(ROOT / "outputs" / "two_view")).strip().strip('"'))
    if not root.is_dir():
        return jsonify({"error": f"Folder not found: {root}"}), 404
    cands = [root] if (root / "points3d.trc").is_file() else \
        sorted((d for d in root.iterdir() if d.is_dir() and (d / "points3d.trc").is_file()),
               key=lambda d: d.stat().st_mtime, reverse=True)
    known = {template_key(t) for t in TEMPLATES.glob("*.json")}
    rows = []
    for d in cands:
        errs, meta = person_errors(d)
        rows.append({"name": meta.get("person_id") or d.name, "path": str(d.resolve()), "errors": errs,
                     "errors_with_template": [e for e in errs if key(e) in known],
                     "mesh": mesh_state(d.resolve()), "created": meta.get("created", "")})
    return jsonify({"root": str(root.resolve()), "folders": rows})


@app.post("/api/mesh/build")
def build():
    folder = folder_arg()
    if mesh_state(folder) == "building":
        return jsonify({"ok": True})
    if not SMPL_PKL.exists():
        return jsonify({"error": f"SMPL model missing: {SMPL_PKL}"}), 400
    (folder / "mesh").mkdir(exist_ok=True)
    BUILDS[str(folder)] = subprocess.Popen([sys.executable, "-u", str(HERE / "p2s_mesh.py"), str(folder)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
    return jsonify({"ok": True})


@app.get("/api/mesh/status")
def build_status():
    folder = folder_arg()
    log = ""
    try:
        log = "\n".join((folder / "mesh" / "log.txt").read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
    except OSError:
        pass
    return jsonify({"state": mesh_state(folder), "log": log})


@app.get("/api/person")
def person():
    folder = folder_arg()
    errs, meta = person_errors(folder)
    res = {}
    try:
        res = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    mesh = None
    if (folder / "mesh" / "mesh.json").exists():
        mesh = json.loads((folder / "mesh" / "mesh.json").read_text(encoding="utf-8"))
    p3d = res.get("p3d") or {}
    return jsonify({
        "name": meta.get("person_id") or folder.name, "path": str(folder), "errors": errs,
        "fps": (res.get("fps") or {}).get("p3d") or (mesh or {}).get("fps") or 30.0,
        "reps": res.get("reps") or [], "depth": (p3d.get("m") or {}).get("squat_depth_knee_deg") or [],
        "videos": [v for v in ("front", "side") if (folder / f"{v}.mp4").exists()],
        "mesh": mesh, "mesh_state": mesh_state(folder)})


@app.get("/api/file")
def file():
    folder = folder_arg()
    name = request.args.get("name", "")
    if name not in FILES_OK or not (folder / name).is_file():
        abort(404)
    return send_file(folder / name, conditional=True,
                     mimetype="application/octet-stream" if name.endswith(".bin") else None)


@app.get("/api/faces.bin")
def faces():
    """SMPL triangles as little-endian uint16 (13776 x 3)."""
    if _faces["bin"] is None:
        import numpy as np
        import smplx
        f = smplx.create(str(SMPL_PKL), model_type="smpl", gender="neutral").faces
        _faces["bin"] = np.asarray(f, dtype="<u2").tobytes()
    return Response(_faces["bin"], mimetype="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/templates")
def templates():
    rows = []
    for t in sorted(TEMPLATES.glob("*.json")):
        try:
            d = json.loads(t.read_text(encoding="utf-8"))
            rows.append({"file": t.name, "name": d.get("mesh_name") or t.stem, "key": template_key(t),
                         "n": len(d.get("saliency") or []), "created": d.get("created_at", "")})
        except (OSError, ValueError):
            rows.append({"file": t.name, "name": t.stem, "key": "", "n": 0, "error": "not readable"})
    return jsonify({"folder": str(TEMPLATES), "templates": rows})


@app.get("/api/templates/<name>")
def template(name):
    p = TEMPLATES / Path(name).name
    if p.suffix != ".json" or not p.is_file():
        abort(404)
    return send_file(p, mimetype="application/json")


@app.post("/api/templates")
def upload_templates():
    saved, errors = [], []
    for f in request.files.getlist("files"):
        try:
            d = json.loads(f.read().decode("utf-8"))
            sal = d.get("saliency")
            if not isinstance(sal, list) or len(sal) != 6890:
                raise ValueError(f"needs 6890 saliency values (got {len(sal) if isinstance(sal, list) else 'none'})")
            name = (d.get("mesh_name") or re.sub(r"_saliency$", "", Path(f.filename).stem)).strip()
            d["mesh_name"] = name
            out = TEMPLATES / (re.sub(r"[^a-z0-9_\-]+", "_", name.lower()).strip("_") + "_saliency.json")
            out.write_text(json.dumps(d), encoding="utf-8")
            saved.append(out.name)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.filename}: {e}")
    return jsonify({"saved": saved, "errors": errors}), (200 if saved or not errors else 400)


@app.delete("/api/templates/<name>")
def delete_template(name):
    p = TEMPLATES / Path(name).name
    if p.suffix == ".json" and p.is_file():
        p.unlink()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("MESH_PORT", "5070"))
    print(f"Mesh viewer: http://127.0.0.1:{port}   (templates folder: {TEMPLATES})")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
