"""
check_setup.py — run this from the repo root before pushing to your Space.
Verifies every required file is where app.py expects it, and tells you
exactly what's missing / misnamed rather than letting you wait for a failed
Space build to find out.

Usage:
    python check_setup.py
"""
import os
import sys

ALPHAPOSE_DIR = os.environ.get("ALPHAPOSE_DIR", "./AlphaPose")
MOTIONBERT_DIR = os.environ.get("MOTIONBERT_DIR", "./MotionBERT")

REQUIRED = [
    # (path, what it is, why it matters)
    (os.path.join(ALPHAPOSE_DIR, "scripts/demo_inference.py"),
     "AlphaPose source code", "clone the AlphaPose repo into ./AlphaPose"),
    (os.path.join(ALPHAPOSE_DIR, "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"),
     "AlphaPose Halpe-26 config", "comes with the AlphaPose clone"),
    (os.path.join(ALPHAPOSE_DIR, "pretrained_models/halpe26_fast_res50_256x192.pth"),
     "AlphaPose Halpe-26 weights", "your downloaded 'halpe26' file, renamed exactly as above"),
    (os.path.join(ALPHAPOSE_DIR, "setup.py"),
     "AlphaPose setup.py", "needed for the '-e ./AlphaPose' editable install to compile it"),

    (os.path.join(MOTIONBERT_DIR, "infer_wild.py"),
     "MotionBERT source code", "clone the MotionBERT repo into ./MotionBERT"),
    (os.path.join(MOTIONBERT_DIR, "infer_wild_mesh.py"),
     "MotionBERT source code", "comes with the MotionBERT clone"),
    (os.path.join(MOTIONBERT_DIR, "requirements.txt"),
     "MotionBERT's own requirements.txt", "referenced by this Space's top-level requirements.txt"),
    (os.path.join(MOTIONBERT_DIR, "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"),
     "MotionBERT 3D-pose checkpoint", "your 'best_epoch_pose3d' file, renamed to best_epoch.bin at this exact path"),
    (os.path.join(MOTIONBERT_DIR, "checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin"),
     "MotionBERT mesh checkpoint", "your 'best_epoch_mesh' file, renamed to best_epoch.bin at this exact path"),
    (os.path.join(MOTIONBERT_DIR, "data/mesh/smpl/SMPL_NEUTRAL.pkl"),
     "SMPL body model", "your downloaded SMPL file, renamed to SMPL_NEUTRAL.pkl at this exact path"),

    ("./yolo11s.pt", "YOLOv11 detector weights",
     "your downloaded yolo11s.pt, placed in the repo root"),
]


def main():
    missing = []
    for path, what, hint in REQUIRED:
        ok = os.path.exists(path)
        status = "OK  " if ok else "MISS"
        print(f"[{status}] {path}   ({what})")
        if not ok:
            missing.append((path, hint))

    print()
    if missing:
        print(f"{len(missing)} required file(s) missing:\n")
        for path, hint in missing:
            print(f"  - {path}\n      -> {hint}")
        print("\nFix these before pushing — a Space build will fail on the "
              "same missing files, just slower to find out.")
        sys.exit(1)
    else:
        print("All required files present. Remember: this only checks "
              "*paths*, not that the weight files themselves are valid/"
              "uncorrupted, and it doesn't run the actual C++ build step "
              "for AlphaPose — that only happens during the Space build.")


if __name__ == "__main__":
    main()
