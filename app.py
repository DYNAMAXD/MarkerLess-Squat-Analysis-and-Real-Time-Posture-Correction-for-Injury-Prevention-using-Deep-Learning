"""
app.py — Video -> AlphaPose (2D) -> MotionBERT (3D pose + mesh) pipeline.

WHAT THIS FILE DOES
--------------------
1. Takes an input video.
2. Calls AlphaPose's own `scripts/demo_inference.py` (as a subprocess) to
   extract 2D keypoints in Halpe-26 format -> alphapose-results.json
3. Calls MotionBERT's own `infer_wild.py` (as a subprocess) to lift the 2D
   keypoints to 3D (H36M 17-joint format) -> X3D.npy
4. Converts X3D.npy into a tidy, frame-by-frame CSV of every joint's (x,y,z).
5. Calls MotionBERT's own `infer_wild_mesh.py` (as a subprocess) to produce
   the SMPL mesh sequence (vertices + rendered overlay video), using the
   X3D.npy from step 3 as the root-trajectory reference.
6. Exports the mesh vertices as a sequence of .obj files (one per frame,
   or every Nth frame) for use in Blender/MeshLab/etc.

WHY SUBPROCESS CALLS INSTEAD OF RE-IMPLEMENTING THE MODELS INLINE
-------------------------------------------------------------------
AlphaPose and MotionBERT are large research codebases (custom CUDA/C++ ops,
specific coordinate normalization, specific checkpoint loading logic, config
plumbing). Re-typing their internals from memory into "one file" would very
likely silently produce wrong joint numbers. Calling their own, maintained
CLI entry points is the only way to guarantee this matches the papers'
released code. This script is the "glue": it does real work (running the
pipeline end-to-end, producing your CSV + mesh files, and giving you a
Gradio UI for a Hugging Face Space), while delegating model internals to
the upstream, unmodified repos.

DIRECTORY LAYOUT THIS SCRIPT EXPECTS (see MODELS.md for exact download steps)
-------------------------------------------------------------------------------
/app/AlphaPose/                        <- git clone of MVIG-SJTU/AlphaPose, built
    pretrained_models/halpe26_fast_res50_256x192.pth
    pretrained_models/yolov3-spp.weights   (or yolox weights, see MODELS.md)
    configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml

/app/MotionBERT/                       <- git clone of Walter0807/MotionBERT
    checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
    checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin
    data/mesh/smpl/SMPL_NEUTRAL.pkl    <- YOU must supply this (SMPL license)

Both paths are overridable with the ALPHAPOSE_DIR / MOTIONBERT_DIR env vars.

USAGE
-----
CLI (no GUI):
    python app.py --video input.mp4 --out_dir outputs/

As a Hugging Face Space (Gradio UI), just run with no --video:
    python app.py
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration — override via environment variables in your Dockerfile/Space
# --------------------------------------------------------------------------
ALPHAPOSE_DIR = os.environ.get("ALPHAPOSE_DIR", "/app/AlphaPose")
MOTIONBERT_DIR = os.environ.get("MOTIONBERT_DIR", "/app/MotionBERT")

ALPHAPOSE_CFG = os.environ.get(
    "ALPHAPOSE_CFG",
    os.path.join(ALPHAPOSE_DIR, "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"),
)
ALPHAPOSE_CKPT = os.environ.get(
    "ALPHAPOSE_CKPT",
    os.path.join(ALPHAPOSE_DIR, "pretrained_models/halpe26_fast_res50_256x192.pth"),
)

MOTIONBERT_POSE3D_CKPT = os.environ.get(
    "MOTIONBERT_POSE3D_CKPT",
    os.path.join(MOTIONBERT_DIR, "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"),
)
MOTIONBERT_MESH_CKPT = os.environ.get(
    "MOTIONBERT_MESH_CKPT",
    os.path.join(MOTIONBERT_DIR, "checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin"),
)

# H36M 17-joint order used by MotionBERT's 3D pose / X3D.npy output.
# (See Walter0807/MotionBERT docs: "The model uses 17 body keypoints, H36M format".)
H36M_JOINT_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck_base", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

# Halpe-26 joint order produced by AlphaPose (body 0-25; first 17 correspond
# roughly to COCO ordering, 17-25 are head/feet extras). Kept here so the raw
# 2D keypoints can also be dumped to CSV if you want them alongside the 3D.
HALPE26_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip", "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe", "left_heel", "right_heel",
]


def _run(cmd, cwd=None):
    print("[app.py] running:", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


# --------------------------------------------------------------------------
# Optional: YOLOv11 (Ultralytics) as the person detector, instead of
# AlphaPose's bundled YOLOv3-SPP. This avoids downloading yolov3-spp.weights
# entirely and lets AlphaPose skip running its own detector, via its
# documented `--detfile <json>` / FileDetectionLoader path ("Detecting
# results from other detectors are also supported as a file input" —
# AlphaPose paper, System section).
#
# Ultralytics YOLOv11 weights are AGPL-3.0 licensed (or require a commercial
# Ultralytics license) — check that fits your use case before shipping this.
# --------------------------------------------------------------------------
def run_yolov11_person_detection(video_path: str, work_dir: str,
                                  model_name: str = "yolo11n.pt",
                                  conf: float = 0.4) -> str:
    """
    Runs Ultralytics YOLOv11 over every frame of the video, keeps only
    person detections (COCO class 0), and writes them as a JSON file in
    the COCO-results format AlphaPose's FileDetectionLoader consumes:
        [{"image_id": <frame_index>, "category_id": 1,
          "bbox": [x, y, w, h], "score": <float>}, ...]

    VERIFY BEFORE TRUSTING SILENTLY: AlphaPose's exact expectation for
    `image_id` (0-based frame index vs. an "AlphaPose_<n>.jpg"-style name)
    has varied across commits of alphapose/utils/file_detector.py. Quick
    sanity check: run AlphaPose once on a couple of frames with its own
    `--detector yolo`, inspect the intermediate detection json it logs/
    caches, and match this function's `image_id` field to that convention
    before relying on this for a full run.
    """
    from ultralytics import YOLO
    import cv2

    os.makedirs(work_dir, exist_ok=True)
    model = YOLO(model_name)  # auto-downloads the weight file on first use

    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, classes=[0], verbose=False)  # class 0 = person
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            detections.append({
                "image_id": frame_idx,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })
        frame_idx += 1
    cap.release()

    det_path = os.path.join(work_dir, "yolov11_detections.json")
    with open(det_path, "w") as f:
        json.dump(detections, f)
    print(f"[app.py] YOLOv11 found {len(detections)} person boxes across {frame_idx} frames -> {det_path}")
    return det_path


# --------------------------------------------------------------------------
# Step 1: AlphaPose — video -> 2D keypoints JSON (Halpe-26)
# --------------------------------------------------------------------------
def run_alphapose(video_path: str, work_dir: str, detfile_path: str = None) -> str:
    os.makedirs(work_dir, exist_ok=True)
    for required in (ALPHAPOSE_CFG, ALPHAPOSE_CKPT):
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"Missing AlphaPose file: {required}\n"
                f"See MODELS.md for exact download instructions."
            )

    demo_script = os.path.join(ALPHAPOSE_DIR, "scripts/demo_inference.py")
    cmd = [
        sys.executable, demo_script,
        "--cfg", ALPHAPOSE_CFG,
        "--checkpoint", ALPHAPOSE_CKPT,
        "--video", video_path,
        "--outdir", work_dir,
        "--sp",            # single-process mode, safer inside a container
        "--pose_track",    # keeps a single consistent person ID across frames
        "--save_video",
    ]
    if detfile_path:
        # Use pre-computed (e.g. YOLOv11) boxes instead of AlphaPose's own
        # bundled detector — no --detector/yolov3-spp.weights needed at all.
        cmd += ["--detfile", detfile_path]
    else:
        cmd += ["--detector", "yolo"]  # AlphaPose's own YOLOv3-SPP
    _run(cmd, cwd=ALPHAPOSE_DIR)

    json_path = os.path.join(work_dir, "alphapose-results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"AlphaPose did not produce {json_path}. Check the AlphaPose logs above."
        )
    return json_path


# --------------------------------------------------------------------------
# Step 2: MotionBERT — 2D keypoints -> 3D pose (H36M, 17 joints)
# --------------------------------------------------------------------------
def run_motionbert_pose3d(video_path: str, json_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(MOTIONBERT_POSE3D_CKPT):
        raise FileNotFoundError(
            f"Missing MotionBERT 3D-pose checkpoint: {MOTIONBERT_POSE3D_CKPT}\n"
            f"See MODELS.md."
        )

    script = os.path.join(MOTIONBERT_DIR, "infer_wild.py")
    cmd = [
        sys.executable, script,
        "--vid_path", video_path,
        "--json_path", json_path,
        "--out_path", out_dir,
    ]
    _run(cmd, cwd=MOTIONBERT_DIR)

    npy_path = os.path.join(out_dir, "X3D.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"MotionBERT did not produce {npy_path}.")
    return npy_path


def pose3d_npy_to_csv(npy_path: str, csv_path: str) -> str:
    """
    Converts MotionBERT's X3D.npy (shape [T, 17, 3]) into a tidy,
    frame-by-frame CSV: one row per (frame, joint).
    """
    poses = np.load(npy_path)  # [T, J, 3]
    if poses.ndim != 3 or poses.shape[1] != len(H36M_JOINT_NAMES):
        print(
            f"[app.py] WARNING: unexpected pose array shape {poses.shape}; "
            f"expected [T, {len(H36M_JOINT_NAMES)}, 3]. Writing generic joint indices."
        )
        joint_names = [f"joint_{i}" for i in range(poses.shape[1])]
    else:
        joint_names = H36M_JOINT_NAMES

    rows = []
    for frame_idx in range(poses.shape[0]):
        for j, jname in enumerate(joint_names):
            x, y, z = poses[frame_idx, j, :3]
            rows.append({"frame": frame_idx, "joint_id": j, "joint_name": jname,
                         "x": float(x), "y": float(y), "z": float(z)})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Also write a wide version (one row per frame, columns per joint) since
    # that's often more convenient for spreadsheets / quick plotting.
    wide = df.pivot(index="frame", columns="joint_name", values=["x", "y", "z"])
    wide.columns = [f"{jname}_{axis}" for axis, jname in wide.columns]
    wide_path = csv_path.replace(".csv", "_wide.csv")
    wide.to_csv(wide_path)

    print(f"[app.py] wrote {csv_path} and {wide_path}")
    return csv_path


# --------------------------------------------------------------------------
# Step 3: MotionBERT — mesh (SMPL) recovery
# --------------------------------------------------------------------------
def run_motionbert_mesh(video_path: str, json_path: str, out_dir: str, ref_3d_motion_path: str = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(MOTIONBERT_MESH_CKPT):
        raise FileNotFoundError(
            f"Missing MotionBERT mesh checkpoint: {MOTIONBERT_MESH_CKPT}\nSee MODELS.md."
        )
    smpl_model = os.path.join(MOTIONBERT_DIR, "data/mesh/smpl/SMPL_NEUTRAL.pkl")
    if not os.path.exists(smpl_model):
        raise FileNotFoundError(
            "Missing SMPL body model at "
            f"{smpl_model}\n"
            "This file is NOT redistributable — you must register at "
            "https://smpl.is.tue.mpg.de/ and download it yourself. See MODELS.md."
        )

    script = os.path.join(MOTIONBERT_DIR, "infer_wild_mesh.py")
    cmd = [
        sys.executable, script,
        "--vid_path", video_path,
        "--json_path", json_path,
        "--out_path", out_dir,
    ]
    if ref_3d_motion_path and os.path.exists(ref_3d_motion_path):
        cmd += ["--ref_3d_motion_path", ref_3d_motion_path]
    _run(cmd, cwd=MOTIONBERT_DIR)
    return out_dir


def export_mesh_frames_as_obj(mesh_out_dir: str, obj_dir: str, stride: int = 1) -> str:
    """
    Best-effort export of the mesh sequence to per-frame .obj files.
    infer_wild_mesh.py saves its raw vertex/face data as an .npz/.npy inside
    mesh_out_dir; the exact key name has changed across MotionBERT commits,
    so this function tries the common ones and tells you what it found if
    none match, rather than failing silently.
    """
    os.makedirs(obj_dir, exist_ok=True)
    candidates = glob.glob(os.path.join(mesh_out_dir, "*.npz")) + \
                 glob.glob(os.path.join(mesh_out_dir, "*.npy"))
    if not candidates:
        print(f"[app.py] No .npz/.npy mesh output found in {mesh_out_dir}; "
              f"skipping .obj export. Check the rendered mesh video there instead.")
        return obj_dir

    for cand in candidates:
        try:
            data = np.load(cand, allow_pickle=True)
        except Exception as e:
            continue
        verts = None
        faces = None
        if isinstance(data, np.lib.npyio.NpzFile):
            for key in ("verts", "vertices", "smpl_verts", "pred_verts"):
                if key in data:
                    verts = data[key]
                    break
            for key in ("faces", "smpl_faces"):
                if key in data:
                    faces = data[key]
                    break
        else:
            verts = data  # plain .npy of vertices

        if verts is None:
            continue

        # Fall back to the SMPL face topology shipped with MotionBERT if the
        # output file didn't include faces explicitly.
        if faces is None:
            faces_npy = os.path.join(MOTIONBERT_DIR, "data/mesh/smpl_faces.npy")
            if os.path.exists(faces_npy):
                faces = np.load(faces_npy)
            else:
                print(f"[app.py] Found vertices in {cand} but no face topology; "
                      f"exporting a point cloud .obj instead of a full mesh.")

        verts = np.asarray(verts)
        if verts.ndim == 2:
            verts = verts[None, ...]  # single frame -> add time dim

        for t in range(0, verts.shape[0], max(1, stride)):
            obj_path = os.path.join(obj_dir, f"frame_{t:05d}.obj")
            with open(obj_path, "w") as f:
                for v in verts[t]:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                if faces is not None:
                    for face in faces:
                        # OBJ face indices are 1-based
                        f.write("f " + " ".join(str(int(idx) + 1) for idx in face) + "\n")
        print(f"[app.py] exported {verts.shape[0]} mesh frames from {cand} -> {obj_dir}")
        return obj_dir

    print(f"[app.py] Found {len(candidates)} array file(s) in {mesh_out_dir} but "
          f"none matched expected vertex keys. Inspect them manually: {candidates}")
    return obj_dir


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def run_pipeline(video_path: str, out_dir: str, do_mesh: bool = True, obj_stride: int = 1,
                  use_yolov11: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    ap_dir = os.path.join(out_dir, "alphapose")
    mb_dir = os.path.join(out_dir, "motionbert_pose3d")
    mesh_dir = os.path.join(out_dir, "motionbert_mesh")
    obj_dir = os.path.join(out_dir, "mesh_obj")

    detfile_path = None
    if use_yolov11:
        yolo_dir = os.path.join(out_dir, "yolov11")
        detfile_path = run_yolov11_person_detection(video_path, yolo_dir)

    json_path = run_alphapose(video_path, ap_dir, detfile_path=detfile_path)
    npy_path = run_motionbert_pose3d(video_path, json_path, mb_dir)
    csv_path = pose3d_npy_to_csv(npy_path, os.path.join(out_dir, "joints_3d.csv"))

    mesh_result_dir = None
    obj_result_dir = None
    if do_mesh:
        mesh_result_dir = run_motionbert_mesh(video_path, json_path, mesh_dir, ref_3d_motion_path=npy_path)
        obj_result_dir = export_mesh_frames_as_obj(mesh_result_dir, obj_dir, stride=obj_stride)

    return {
        "keypoints_2d_json": json_path,
        "pose3d_npy": npy_path,
        "joints_csv": csv_path,
        "joints_csv_wide": csv_path.replace(".csv", "_wide.csv"),
        "mesh_dir": mesh_result_dir,
        "mesh_obj_dir": obj_result_dir,
    }


# --------------------------------------------------------------------------
# Entry points: CLI and Gradio (for a Hugging Face Space)
# --------------------------------------------------------------------------
def build_gradio_app():
    import gradio as gr
    import tempfile
    try:
        import spaces  # only present on HF ZeroGPU Spaces
        gpu_decorator = spaces.GPU(duration=300)  # seconds; raise if your videos are long
    except ImportError:
        gpu_decorator = lambda f: f  # no-op locally / on non-ZeroGPU Spaces

    @gpu_decorator
    def _infer(video_file, produce_mesh, use_yolov11):
        work_dir = tempfile.mkdtemp(prefix="pose_pipeline_")
        try:
            result = run_pipeline(video_file, work_dir, do_mesh=produce_mesh, use_yolov11=use_yolov11)
        except Exception as e:
            raise gr.Error(f"Pipeline failed: {e}")

        mesh_video = None
        if result["mesh_dir"]:
            vids = glob.glob(os.path.join(result["mesh_dir"], "*.mp4"))
            mesh_video = vids[0] if vids else None

        return result["joints_csv"], result["joints_csv_wide"], mesh_video

    with gr.Blocks(title="AlphaPose + MotionBERT: Video -> 3D Joints & Mesh") as demo:
        gr.Markdown(
            "# Video -> 2D pose (AlphaPose) -> 3D pose + mesh (MotionBERT)\n"
            "Upload a video with a single, clearly visible person. "
            "Processing is slow on CPU — a GPU Space is strongly recommended."
        )
        with gr.Row():
            video_in = gr.Video(label="Input video")
        mesh_toggle = gr.Checkbox(value=True, label="Also compute SMPL mesh (slower)")
        yolo_toggle = gr.Checkbox(value=False, label="Use YOLOv11 detector instead of AlphaPose's built-in YOLOv3")
        run_btn = gr.Button("Run pipeline", variant="primary")
        with gr.Row():
            csv_out = gr.File(label="Joints CSV (long format: frame, joint, x, y, z)")
            csv_wide_out = gr.File(label="Joints CSV (wide format: one row per frame)")
        mesh_video_out = gr.Video(label="Mesh overlay video (if computed)")

        run_btn.click(_infer, inputs=[video_in, mesh_toggle, yolo_toggle],
                       outputs=[csv_out, csv_wide_out, mesh_video_out])
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=None, help="Path to input video. Omit to launch the Gradio UI instead.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to write all outputs to.")
    parser.add_argument("--no_mesh", action="store_true", help="Skip the (slower) SMPL mesh step; only produce 3D joint CSV.")
    parser.add_argument("--obj_stride", type=int, default=1, help="Export every Nth frame as .obj (1 = every frame).")
    parser.add_argument("--use_yolov11", action="store_true",
                         help="Use Ultralytics YOLOv11 for person detection instead of AlphaPose's built-in YOLOv3-SPP.")
    args = parser.parse_args()

    if args.video:
        result = run_pipeline(args.video, args.out_dir, do_mesh=not args.no_mesh,
                               obj_stride=args.obj_stride, use_yolov11=args.use_yolov11)
        print(json.dumps(result, indent=2))
    else:
        demo = build_gradio_app()
        # show_api=False skips building the /api docs schema entirely — the
        # exact code path that trips the pydantic/gradio_client schema bug
        # above. Cheap insurance even with the requirements.txt pins in place.
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                    show_api=False)


if __name__ == "__main__":
    main()