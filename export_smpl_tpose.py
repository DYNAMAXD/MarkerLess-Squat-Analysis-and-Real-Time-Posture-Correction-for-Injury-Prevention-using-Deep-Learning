"""
export_smpl_tpose.py

Exports the neutral, zero-pose ("T-pose") SMPL_NEUTRAL mesh as a plain .obj
file, so it can be loaded into mesh_saliency_painter.html for painting
per-vertex saliency values.

Usage (run from the same root where your AlphaPose/ and MotionBERT/ folders
already live, i.e. the layout described at the top of keypoint_pipeline.py):

    python export_smpl_tpose.py --out smpl_neutral_tpose.obj

What it does, in order:

  1. Tries your repo's own SMPL wrapper (lib.utils.utils_smpl), the same
     module keypoint_pipeline.py already imports get_smpl_faces() from.
     This keeps the exported mesh on the exact vertex ordering / topology
     your mesh pipeline (verts from lift_sequence_to_mesh) uses, which
     matters because the painter's saliency array is just a flat list of
     6890 values in vertex order -- it has to be the same order every time.
  2. Falls back to the generic `smplx` pip package pointed at the same
     SMPL_NEUTRAL.pkl if step 1 doesn't work in your environment.

NOTE: `lib.utils.utils_smpl`'s exact class/function names vary a bit
between MotionBERT forks. If try_repo_smpl() below fails with an
ImportError/AttributeError, open that file and adjust the import and the
SMPL(...) call to match what's actually defined there -- the shape you're
after is "give it zero pose + zero shape, get back (6890, 3) vertices".
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MOTIONBERT_DIR = ROOT / "MotionBERT"
SMPL_DATA_DIR = MOTIONBERT_DIR / "data" / "mesh"


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ face indices are 1-based
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def try_repo_smpl():
    """Uses your own repo's SMPL wrapper, so the vertex order matches what
    lift_sequence_to_mesh() produces elsewhere in your pipeline."""
    if str(MOTIONBERT_DIR) not in sys.path:
        sys.path.insert(0, str(MOTIONBERT_DIR))

    from lib.utils.utils_smpl import get_smpl_faces, SMPL  # adjust names if your file differs

    faces = np.asarray(get_smpl_faces())
    smpl = SMPL(str(SMPL_DATA_DIR), batch_size=1)

    zero_pose = torch.zeros(1, 72)
    zero_shape = torch.zeros(1, 10)
    with torch.no_grad():
        out = smpl(
            betas=zero_shape,
            body_pose=zero_pose[:, 3:],
            global_orient=zero_pose[:, :3],
        )
        verts = out.vertices[0].cpu().numpy() if hasattr(out, "vertices") else out[0].cpu().numpy()
    return verts, faces


def try_smplx():
    """Generic fallback via the `smplx` pip package (pip install smplx)."""
    import smplx

    model = smplx.create(
        model_path=str(SMPL_DATA_DIR),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
    )
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
        )
    verts = out.vertices[0].cpu().numpy()
    faces = np.asarray(model.faces)
    return verts, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smpl_neutral_tpose.obj", help="Output .obj path")
    args = ap.parse_args()

    try:
        verts, faces = try_repo_smpl()
        print("Loaded mesh via lib.utils.utils_smpl (your repo's own SMPL wrapper).")
    except Exception as e:
        print(f"Repo SMPL wrapper failed ({e!r}); falling back to the smplx package...")
        verts, faces = try_smplx()
        print("Loaded mesh via smplx.")

    write_obj(args.out, verts, faces)
    print(f"Wrote {len(verts)} vertices / {len(faces)} faces to {args.out}")
    print("Load this file into mesh_saliency_painter.html to start painting.")


if __name__ == "__main__":
    main()
