"""
build_space.py -- assembles the folder to upload to the Hugging Face ZeroGPU Space (Gradio; default hf_space/upload/,
git-ignored):
only what the final product needs to run there - the app and its page, the preprocessing scripts, the error model
and its code, the MotionBERT mesh model + SMPL files, the painted saliency templates and a ready calibration.

    python hf_space/build_space.py [--out <folder>] [--model <export folder>]

Needs the trained model export (default final_product/squat_model_export/, a copy of
research/results/04_final/export/) and the MotionBERT mesh checkpoint + SMPL files (Git LFS in this repo).
Standard library only; any Python 3.11 works.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# (source relative to the repo, destination relative to the Space, glob or None = the file / whole folder)
ITEMS = [
    ("hf_space/app.py", "app.py", None),
    ("hf_space/SPACE_README.md", "README.md", None),
    ("hf_space/space.gitattributes", ".gitattributes", None),
    ("hf_space/requirements.txt", "requirements.txt", None),
    ("hf_space/packages.txt", "packages.txt", None),
    ("hf_space/presets", "presets", None),
    ("final_product", "final_product", ["product_app.py", "product_pipeline.py", "product_predictions.py",
                                        "product_content.json", "web/product.html", "web/product.js",
                                        "web/vendor/three.min.js", "web/vendor/fonts/*"]),
    ("two_view", "two_view", ["run_pose2sim.py", "p2s_sync.py", "p2s_extrinsics.py", "p2s_biomech.py", "p2s_mesh.py",
                              "rep_fusion.py", "sync_clap.py", "web_results.py"]),
    ("research/squat_research", "research/squat_research", ["*.py"]),
    ("keypoint_pipeline.py", "keypoint_pipeline.py", None),
    ("MotionBERT", "MotionBERT", ["LICENSE", "lib/**/*.py", "configs/mesh/MB_ft_pw3d.yaml",
                                  "checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin",
                                  "data/mesh/SMPL_NEUTRAL.pkl", "data/mesh/smpl_mean_params.npz",
                                  "data/mesh/J_regressor_extra.npy", "data/mesh/J_regressor_h36m_correct.npy"]),
    ("saliency_maps", "saliency_maps", ["*.json"]),
]
MODEL_FILES = ["model.pt", "config.json", "norm_stats.json", "model_card.md"]


def copy_item(src, dst, patterns, out):
    s, d = ROOT / src, out / dst
    if patterns is None:
        if not s.exists():
            raise SystemExit(f"missing: {s}")
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        return
    for pat in patterns:
        hits = [p for p in s.glob(pat) if p.is_file() and "__pycache__" not in p.parts]
        if not hits:
            raise SystemExit(f"missing: {s / pat}")
        for p in hits:
            t = d / p.relative_to(s)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "upload"), help="output folder (emptied first)")
    ap.add_argument("--model", default=str(ROOT / "final_product" / "squat_model_export"), help="model export folder")
    a = ap.parse_args()
    out, model = Path(a.out).resolve(), Path(a.model).resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        raise SystemExit("--out must be a new folder, not the repo itself")
    missing = [f for f in MODEL_FILES[:3] if not (model / f).exists()]
    if missing:
        raise SystemExit(f"model export incomplete ({', '.join(missing)} missing in {model}). Copy "
                         "research/results/04_final/export/ to final_product/squat_model_export/ first.")
    lfs = ROOT / "MotionBERT" / "data" / "mesh" / "SMPL_NEUTRAL.pkl"
    if lfs.exists() and lfs.stat().st_size < 10_000:
        raise SystemExit("MotionBERT / SMPL files are Git LFS pointers - run `git lfs pull` first.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for src, dst, pats in ITEMS:
        copy_item(src, dst, pats, out)
    md = out / "models" / "squat_model"   # the model the Space uses (see models/README.md: how to change it)
    md.mkdir(parents=True)
    for f in MODEL_FILES:
        if (model / f).exists():
            shutil.copy2(model / f, md / f)
    shutil.copy2(HERE / "models_README.md", out / "models" / "README.md")
    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    big = sorted((p for p in files if p.stat().st_size > 10e6), key=lambda p: -p.stat().st_size)
    print(f"Space folder: {out}\n  {len(files)} files, {size / 1e6:.0f} MB")
    for p in big:
        print(f"  large (Git LFS on the Hub): {p.relative_to(out)}  {p.stat().st_size / 1e6:.0f} MB")
    print("Upload the CONTENTS of this folder to a new Gradio Space with ZeroGPU (see hf_space/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
