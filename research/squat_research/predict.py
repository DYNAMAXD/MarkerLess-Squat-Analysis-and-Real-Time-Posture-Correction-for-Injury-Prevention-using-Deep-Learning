"""
predict.py - runs an exported squat model on one preprocessed recording / product session and writes predictions.json
next to its results.json (the format final_product/web/product.js reads).

    C:\\SquatModelVenv\\Scripts\\python.exe -m squat_research.predict <session folder> --model <export folder>
    (PYTHONPATH = research/, the folder that holds squat_research/; final_product/product_pipeline.py does this)

Per rep: label, confidence, class probabilities, alpha (64 frame-attention weights) and alpha_frames (alpha on the
rep's real frames: 10-frame windows, Gaussian sigma 5 frames, peak 1). intensity = alpha_frames x confidence over the
whole timeline (0 outside reps and on reps predicted 'correct'); the 3D body is coloured template[label] x intensity.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="preprocessed recording / session folder (points3d.trc, advancedangles.csv, reps.csv, results.json)")
    ap.add_argument("--model", required=True, help="export folder (model.pt, config.json, norm_stats.json)")
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    from .engine import predict_recording
    pred = predict_recording(a.folder, a.model)
    for r in pred["reps"]:
        r.pop("flex", None)                       # only used by the research notebooks
    out = Path(a.folder) / "predictions.json"
    out.write_text(json.dumps(pred), encoding="utf-8")
    ok = [r for r in pred["reps"] if r["valid"]]
    print(f"{len(pred['reps'])} reps ({len(ok)} analysed) with {pred['model_name']} in {time.time() - t0:.1f} s -> {out}")
    for r in pred["reps"]:
        print(f"  rep {r['rep']:2d} {r['start_s']:7.2f}-{r['end_s']:7.2f} s  " +
              (f"{r['label']:<20} {r['confidence']:.2f}{'  (uncertain)' if r['uncertain'] else ''}" if r["valid"]
               else f"not analysed: {r.get('reason')}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
