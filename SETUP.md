# SETUP — exact file tree to push to your Hugging Face Space

Your 5 downloaded files are the model *weights*. You still need the
AlphaPose and MotionBERT *source code* (their Python packages/scripts) —
weights alone don't do anything without the code that loads them.

## Final tree your Space repo needs

```
your-space/
├── README.md                (provided — has the sdk: gradio header)
├── packages.txt              (provided)
├── pre-requirements.txt      (provided)
├── requirements.txt          (provided)
├── app.py                    (provided — paths already point at ./AlphaPose, ./MotionBERT)
├── check_setup.py            (provided — run this before you push)
│
├── AlphaPose/                 <- clone https://github.com/MVIG-SJTU/AlphaPose here, full source
│   └── pretrained_models/
│       └── halpe26_fast_res50_256x192.pth      <- YOUR downloaded file, renamed/placed here
│
└── MotionBERT/                <- clone https://github.com/Walter0807/MotionBERT here, full source
    ├── checkpoint/
    │   ├── pose3d/FT_MB_lite_MB_ft_h36m_global_lite/
    │   │   └── best_epoch.bin                   <- YOUR "best_epoch_pose3d" file, renamed here
    │   └── mesh/FT_MB_release_MB_ft_pw3d/
    │       └── best_epoch.bin                   <- YOUR "best_epoch_mesh" file, renamed here
    └── data/mesh/smpl/
        └── SMPL_NEUTRAL.pkl                     <- YOUR downloaded file, renamed here
```

Note: `yolo11s.pt` doesn't need a fixed folder — `app.py`'s
`run_yolov11_person_detection()` just needs the filename passed to
`YOLO(model_name)`; easiest is to drop it in the repo root and set
`model_name="yolo11s.pt"` (edit the default in `app.py`, it currently
defaults to the smaller `yolo11n.pt` — swap since you downloaded `s`).

## Step by step

1. **Locally** (not inside the Space repo yet):
   ```
   git clone https://github.com/MVIG-SJTU/AlphaPose.git
   git clone https://github.com/Walter0807/MotionBERT.git
   ```
2. Copy your 5 downloaded weight files into the exact paths shown above
   (rename `best_epoch_pose3d` → `best_epoch.bin` etc. — the filenames
   matter, `app.py` and the two repos' own scripts look for them by name).
3. Copy `yolo11s.pt` to the repo root.
4. Clone your (empty) HF Space repo locally:
   ```
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   ```
5. Move `AlphaPose/`, `MotionBERT/`, `yolo11s.pt`, and all six provided
   files (`README.md`, `packages.txt`, `pre-requirements.txt`,
   `requirements.txt`, `app.py`, `check_setup.py`) into that Space repo
   folder.
6. **Git-LFS the big binaries before committing**, or the push will fail:
   ```
   cd <space-name>
   git lfs install
   git lfs track "*.pth" "*.pt" "*.pkl" "*.bin"
   git add .gitattributes
   ```
7. Run the checker (catches path/naming mistakes before you wait for a
   Space build):
   ```
   python check_setup.py
   ```
8. `git add . && git commit -m "AlphaPose + MotionBERT space" && git push`
9. Watch the Space's **Build logs** tab. The step most likely to need a
   retry is `-e ./AlphaPose` in requirements.txt (the C++ compile) — see
   the troubleshooting note in `pre-requirements.txt` if it fails.

## Total size, roughly

AlphaPose + MotionBERT source: a few MB. Your 5 weight files: ~130MB
(halpe26) + ~40MB (SMPL) + ~20MB (yolo11s) + ~61MB (pose3d) + ~162MB
(mesh) ≈ 400MB+. Comfortably within a normal HF Space's storage, but LFS
step 6 above is not optional at this size.
