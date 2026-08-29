"""
motion_dashboard.py
===================

CSV + Video motion-analysis dashboard using Dash + Plotly.

Install
-------
python -m pip install -U dash plotly pandas numpy opencv-python

Run
---
python motion_dashboard.py

Open
----
http://127.0.0.1:8050

UPLOAD OPTIONS
--------------
There are two ways to provide files.

1. Normal browser upload
   - CSV upload box
   - Video upload box

2. Direct local-path mode
   - Paste the absolute Windows path to the CSV
   - Paste the absolute Windows path to the video

Direct-path mode is especially useful for large videos because it avoids
sending the entire video through the browser as base64.

CSV expected columns
--------------------
frame,
knee_valgus_L,
knee_valgus_R,
head_forward_angle,
squat_depth_knee_deg,
sagittal_flexion_trunk,
hip_angle_L,
hip_angle_R,
hip_flexion_L,
hip_flexion_R,
knee_flexion_L,
knee_flexion_R,
ankle_dorsiflexion_proxy_L,
ankle_dorsiflexion_proxy_R,
lumbar_curvature_proxy

MAIN INTERACTION
----------------
Click any point/line on the Plotly graph.

That frame becomes the selected frame and updates:
    - video frame
    - frame slider
    - vertical graph marker
    - current-frame cards
    - selected-column statistics

Other useful controls
---------------------
- Multiple metric selection
- Raw / smoothed signal
- Rolling smoothing window
- Line width
- Point markers
- Plot height
- Visible x-axis range
- CSV-to-video frame offset

Frame mapping
-------------
video frame = CSV frame + CSV frame offset

For example:
    CSV frame 1 -> video frame 0
Use:
    CSV frame offset = -1
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import dash
from dash import Dash, Input, Output, State, dcc, html, dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================================
# Configuration
# ============================================================================

METRICS = [
    "knee_valgus_L",
    "knee_valgus_R",
    "head_forward_angle",
    "squat_depth_knee_deg",
    "sagittal_flexion_trunk",
    "hip_angle_L",
    "hip_angle_R",
    "hip_flexion_L",
    "hip_flexion_R",
    "knee_flexion_L",
    "knee_flexion_R",
    "ankle_dorsiflexion_proxy_L",
    "ankle_dorsiflexion_proxy_R",
    "lumbar_curvature_proxy",
]

REQUIRED_COLUMNS = ["frame"] + METRICS

UPLOAD_DIR = Path(tempfile.gettempdir()) / "motion_dashboard_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utility functions
# ============================================================================

def fmt(value, digits=2):
    if value is None:
        return "—"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(value):
        return "—"

    if digits == 0:
        return f"{value:,.0f}"

    return f"{value:,.{digits}f}"


def make_card(title, value, subtitle=""):
    return html.Div(
        [
            html.Div(title, className="card-title"),
            html.Div(value, className="card-value"),
            html.Div(subtitle, className="card-subtitle"),
        ],
        className="metric-card",
    )


def empty_figure(message="Upload CSV and video to begin"):
    fig = go.Figure()

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#0B0B0B",
        plot_bgcolor="#0B0B0B",
        height=620,
        margin=dict(l=65, r=30, t=70, b=60),
        title=message,
        xaxis=dict(
            title=dict(
                text="Frame",
                font=dict(color="#F5F5F5"),
            ),
            color="#F5F5F5",
            gridcolor="#333333",
            zerolinecolor="#333333",
        ),
        yaxis=dict(
            title=dict(
                text="Metric value",
                font=dict(color="#F5F5F5"),
            ),
            color="#F5F5F5",
            gridcolor="#333333",
            zerolinecolor="#333333",
        ),
        hovermode="closest",
    )

    return fig


def save_upload(contents: str, filename: str, prefix: str) -> str:
    if not contents:
        raise ValueError("No uploaded file data was received.")

    if "," not in contents:
        raise ValueError("Invalid upload data.")

    _, encoded = contents.split(",", 1)

    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError(f"Could not decode uploaded file: {exc}") from exc

    safe_name = Path(filename or f"{prefix}_upload").name

    target = UPLOAD_DIR / safe_name

    # Avoid stale clashes when the user uploads files with the same name.
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        i = 1

        while target.exists():
            target = UPLOAD_DIR / f"{stem}_{i}{suffix}"
            i += 1

    target.write_bytes(raw)

    return str(target)


def resolve_path(
    uploaded_path: Optional[str],
    typed_path: Optional[str],
) -> Optional[str]:
    """
    Prefer a directly typed local path when it exists.
    Otherwise use the uploaded server-side path.
    """
    typed_path = (typed_path or "").strip().strip('"')

    if typed_path:
        candidate = Path(typed_path)

        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

    if uploaded_path:
        candidate = Path(uploaded_path)

        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

    return None


def read_csv(csv_path: str) -> pd.DataFrame:
    if not csv_path:
        raise ValueError("CSV path is empty.")

    if not os.path.isfile(csv_path):
        raise ValueError(f"CSV file does not exist:\n{csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV:\n{exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + "\n".join(f"  {c}" for c in missing)
        )

    ordered = REQUIRED_COLUMNS + [
        c for c in df.columns if c not in REQUIRED_COLUMNS
    ]

    df = df[ordered].copy()

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    for col in METRICS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df.dropna(subset=["frame"])
        .sort_values("frame")
        .drop_duplicates(
            subset=["frame"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "CSV contains no usable frame rows."
        )

    return df


def get_video_info(video_path: Optional[str]):
    if not video_path or not os.path.isfile(video_path):
        return {
            "frames": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "frames": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
        }

    info = {
        "frames": int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ),
        "fps": float(
            cap.get(cv2.CAP_PROP_FPS) or 0.0
        ),
        "width": int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        ),
        "height": int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        ),
    }

    cap.release()

    return info


def nearest_csv_frame(
    df: pd.DataFrame,
    requested_frame: float,
) -> float:
    idx = (
        df["frame"] - float(requested_frame)
    ).abs().idxmin()

    return float(
        df.loc[idx, "frame"]
    )


def get_selected_row(
    df: pd.DataFrame,
    requested_frame: float,
) -> pd.Series:
    idx = (
        df["frame"] - float(requested_frame)
    ).abs().idxmin()

    return df.loc[idx]


def read_video_frame(
    video_path: Optional[str],
    csv_frame: float,
    frame_offset: int,
):
    if not video_path:
        return None, None

    info = get_video_info(video_path)

    if info["frames"] <= 0:
        return None, None

    video_index = int(
        round(
            float(csv_frame)
            + int(frame_offset)
        )
    )

    video_index = max(
        0,
        min(
            video_index,
            info["frames"] - 1,
        ),
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return None, None

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        video_index,
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        return None, video_index

    return (
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        ),
        video_index,
    )


def image_to_data_uri(
    image: Optional[np.ndarray],
):
    if image is None:
        return None

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            92,
        ],
    )

    if not ok:
        return None

    encoded_b64 = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    return (
        "data:image/jpeg;base64,"
        + encoded_b64
    )


# ============================================================================
# Plot
# ============================================================================

def make_figure(
    df: Optional[pd.DataFrame],
    selected: list[str],
    smoothing: int,
    plot_options: list[str],
    line_width: float,
    current_frame: Optional[float],
    x_start: Optional[float],
    x_end: Optional[float],
    plot_height: int,
):
    if df is None or df.empty:
        return empty_figure()

    selected = selected or []

    if not selected:
        return empty_figure(
            "Select one or more metrics"
        )

    plot_options = plot_options or []

    show_raw = "raw" in plot_options
    show_smooth = "smooth" in plot_options
    show_markers = "markers" in plot_options

    fig = go.Figure()

    smoothing = max(
        1,
        int(smoothing or 1),
    )

    x = df["frame"]

    for metric in selected:
        y = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

        if show_raw:
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode=(
                        "lines+markers"
                        if show_markers
                        else "lines"
                    ),
                    name=metric,
                    line=dict(
                        width=float(
                            line_width or 2
                        ),
                    ),
                    marker=dict(size=4),
                    customdata=np.asarray(
                        x
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Frame: %{x}<br>"
                        "Value: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

        if show_smooth and smoothing > 1:
            smooth = (
                y.rolling(
                    window=smoothing,
                    center=True,
                    min_periods=max(
                        1,
                        smoothing // 2,
                    ),
                )
                .mean()
            )

            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=smooth,
                    mode="lines",
                    name=f"{metric} • smooth",
                    line=dict(
                        width=max(
                            float(
                                line_width or 2
                            )
                            + 1.0,
                            1.5,
                        ),
                    ),
                    customdata=np.asarray(
                        x
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Frame: %{x}<br>"
                        "Value: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

    if current_frame is not None:
        fig.add_vline(
            x=float(current_frame),
            line_width=2,
            line_dash="dash",
            line_color="#333333",
            annotation_text=(
                f"Frame "
                f"{fmt(current_frame, 0)}"
            ),
            annotation_position="top right",
        )

    frame_min = float(
        df["frame"].min()
    )
    frame_max = float(
        df["frame"].max()
    )

    if x_start is not None:
        try:
            if np.isfinite(float(x_start)):
                frame_min = max(
                    frame_min,
                    float(x_start),
                )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if x_end is not None:
        try:
            if np.isfinite(float(x_end)):
                frame_max = min(
                    frame_max,
                    float(x_end),
                )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if frame_min > frame_max:
        frame_min, frame_max = (
            frame_max,
            frame_min,
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#0B0B0B",
        plot_bgcolor="#0B0B0B",
        height=int(plot_height or 620),
        margin=dict(
            l=65,
            r=30,
            t=75,
            b=60,
        ),
        title=(
            "Joint-angle / motion metrics"
        ),
        hovermode="closest",
        dragmode="zoom",
        clickmode="event+select",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            title=dict(
                text="Frame",
                font=dict(color="#F5F5F5"),
            ),
            range=[
                frame_min,
                frame_max,
            ],
            color="#F5F5F5",
            gridcolor="#333333",
            zerolinecolor="#333333",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        ),

        yaxis=dict(
            title=dict(
                text="Metric value",
                font=dict(color="#F5F5F5"),
            ),
            color="#F5F5F5",
            gridcolor="#333333",
            zerolinecolor="#333333",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
        ),
    )

    return fig


# ============================================================================
# Statistics
# ============================================================================

def make_statistics(
    df: Optional[pd.DataFrame],
    selected: list[str],
    current_frame: Optional[float],
):
    columns = [
        "Metric",
        "Current",
        "Mean",
        "Median",
        "Std",
        "Min",
        "Max",
        "Range",
        "Valid N",
    ]

    if (
        df is None
        or df.empty
        or not selected
    ):
        return []

    row = (
        get_selected_row(
            df,
            current_frame,
        )
        if current_frame is not None
        else None
    )

    records = []

    for metric in selected:
        series = pd.to_numeric(
            df[metric],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            records.append(
                {
                    "Metric": metric,
                    "Current": None,
                    "Mean": None,
                    "Median": None,
                    "Std": None,
                    "Min": None,
                    "Max": None,
                    "Range": None,
                    "Valid N": 0,
                }
            )
            continue

        current = None

        if (
            row is not None
            and pd.notna(row[metric])
        ):
            current = float(
                row[metric]
            )

        records.append(
            {
                "Metric": metric,
                "Current": current,
                "Mean": float(
                    series.mean()
                ),
                "Median": float(
                    series.median()
                ),
                "Std": float(
                    series.std(
                        ddof=1
                    )
                )
                if len(series) > 1
                else 0.0,
                "Min": float(
                    series.min()
                ),
                "Max": float(
                    series.max()
                ),
                "Range": float(
                    series.max()
                    - series.min()
                ),
                "Valid N": int(
                    series.count()
                ),
            }
        )

    return records


def make_cards(
    df: Optional[pd.DataFrame],
    selected: list[str],
    current_frame: Optional[float],
    video_frame: Optional[int],
    frame_offset: int,
):
    if (
        df is None
        or df.empty
        or current_frame is None
    ):
        return [
            make_card(
                "Current frame",
                "—",
            ),
            make_card(
                "Video frame",
                "—",
            ),
            make_card(
                "Metrics selected",
                "0",
            ),
            make_card(
                "Frame average",
                "—",
            ),
            make_card(
                "Frame spread",
                "—",
            ),
            make_card(
                "Frame max",
                "—",
            ),
        ]

    row = get_selected_row(
        df,
        current_frame,
    )

    values = (
        pd.to_numeric(
            row[selected],
            errors="coerce",
        )
        .dropna()
        if selected
        else pd.Series(
            dtype=float
        )
    )

    if len(values):
        frame_avg = values.mean()
        frame_spread = (
            values.max()
            - values.min()
        )
        frame_max = values.max()
    else:
        frame_avg = None
        frame_spread = None
        frame_max = None

    return [
        make_card(
            "Current frame",
            fmt(
                row["frame"],
                0,
            ),
        ),
        make_card(
            "Video frame",
            (
                fmt(
                    video_frame,
                    0,
                )
                if video_frame is not None
                else "—"
            ),
            (
                f"Offset "
                f"{int(frame_offset):+d}"
            ),
        ),
        make_card(
            "Metrics selected",
            str(len(selected)),
        ),
        make_card(
            "Frame average",
            fmt(frame_avg),
        ),
        make_card(
            "Frame spread",
            fmt(frame_spread),
        ),
        make_card(
            "Frame max",
            fmt(frame_max),
        ),
    ]


# ============================================================================
# CSS
# ============================================================================
CUSTOM_CSS = """
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #0B0B0B;
    color: #F5F5F5;
    font-family: Arial, Helvetica, sans-serif;
}

.page {
    max-width: 1700px;
    margin: auto;
    padding: 24px;
    background: #0B0B0B;
    min-height: 100vh;
}

.header {
    margin-bottom: 20px;
}

.header h1 {
    margin: 0 0 8px 0;
    color: #F5F5F5;
    font-size: 30px;
}

.header p {
    margin: 0;
    color: #F5F5F5;
    opacity: 0.75;
}

.layout {
    display: grid;
    grid-template-columns: 360px minmax(0, 1fr);
    gap: 20px;
}

.panel {
    background: #1F1F1F;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 16px;
}

.section-title {
    color: #F5F5F5;
    font-size: 18px;
    font-weight: 700;
    margin: 4px 0 14px 0;
}

.control-title {
    color: #F5F5F5;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-weight: 700;
    margin: 16px 0 7px 0;
}


/* ============================================================
   UPLOAD BOXES
   ============================================================ */

.upload-box {
    border: 1px dashed #333333;
    background: #0B0B0B;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    cursor: pointer;
    margin-bottom: 8px;
    color: #F5F5F5;
}

.upload-box:hover {
    border-color: #F5F5F5;
    background: #1F1F1F;
}

.upload-box button {
    padding: 9px 14px;
    border: 1px solid #333333;
    background: #333333;
    color: #F5F5F5;
    border-radius: 8px;
    cursor: pointer;
}

.upload-box button:hover {
    background: #FF7C00;
}

.file-name {
    font-size: 12px;
    color: #F5F5F5;
    margin: 6px 0 12px 0;
    word-break: break-all;
}


/* ============================================================
   TEXT INPUTS
   ============================================================ */

.path-input {
    width: 100%;
    padding: 9px 10px;
    border-radius: 8px;
    border: 1px solid #333333;
    background: #0B0B0B;
    color: #F5F5F5;
    outline: none;
}

.path-input:focus {
    border-color: #F5F5F5;
    box-shadow: 0 0 0 1px #333333;
}

.path-input::placeholder {
    color: #F5F5F5;
    opacity: 0.5;
}

.path-help {
    color: #F5F5F5;
    opacity: 0.7;
    font-size: 11px;
    margin-top: 5px;
    line-height: 1.4;
}


/* ============================================================
   STATUS
   ============================================================ */

.status {
    background: #0B0B0B;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 12px;
    white-space: pre-wrap;
    line-height: 1.5;
    color: #F5F5F5;
    margin-top: 16px;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.cards {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 15px;
}

.metric-card {
    background: #0B0B0B;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 12px;
    min-height: 86px;
}

.card-title {
    color: #F5F5F5;
    opacity: 0.65;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.card-value {
    color: #F5F5F5;
    font-size: 22px;
    font-weight: 700;
    margin-top: 7px;
}

.card-subtitle {
    color: #F5F5F5;
    opacity: 0.6;
    font-size: 10px;
    margin-top: 3px;
}


/* ============================================================
   FRAME LABEL
   ============================================================ */

.frame-label {
    color: #F5F5F5;
    font-size: 13px;
    font-weight: 700;
    margin: 6px 0;
}


/* ============================================================
   VIDEO
   ============================================================ */

.video-box {
    background: #F5F5F5;
    border: 1px solid #333333;
    border-radius: 10px;
    min-height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.video-box img {
    max-width: 100%;
    max-height: 600px;
    display: block;
}

.frame-placeholder {
    text-align: center;
    padding: 50px;
    color: #0B0B0B;
}


/* ============================================================
   DASH DROPDOWN
   ============================================================ */

.Select-control {
    background-color: #0B0B0B !important;
    border: 1px solid #333333 !important;
    color: #F5F5F5 !important;
}

.Select-control:hover {
    border-color: #F5F5F5 !important;
}

.Select-value-label,
.Select-placeholder,
.Select-input > input {
    color: #F5F5F5 !important;
}

.Select-menu-outer {
    background-color: #0B0B0B !important;
    border: 1px solid #333333 !important;
}

.VirtualizedSelectOption {
    background-color: #0B0B0B !important;
    color: #F5F5F5 !important;
}

.VirtualizedSelectFocusedOption {
    background-color: #1F1F1F !important;
    color: #F5F5F5 !important;
}

.Select-value {
    background-color: #1F1F1F !important;
    border: 1px solid #333333 !important;
    color: #F5F5F5 !important;
}

.Select-value-label {
    color: #F5F5F5 !important;
}

.Select-value-icon {
    color: #F5F5F5 !important;
    border-right: 1px solid #333333 !important;
}

.Select-value-icon:hover {
    background-color: #333333 !important;
    color: #F5F5F5 !important;
}


/* ============================================================
   DASH SLIDERS
   ============================================================ */

.rc-slider {
    margin: 8px 3px 14px 3px;
}

.rc-slider-rail {
    background-color: #333333 !important;
}

.rc-slider-track {
    background-color: #F5F5F5 !important;
}

.rc-slider-handle {
    border: 2px solid #F5F5F5 !important;
    background-color: #0B0B0B !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

.rc-slider-handle:hover,
.rc-slider-handle:active {
    border-color: #F5F5F5 !important;
    box-shadow: 0 0 0 4px rgba(226, 180, 189, 0.35) !important;
}

.rc-slider-mark-text {
    color: #F5F5F5 !important;
}

.rc-slider-tooltip-inner {
    background-color: #F5F5F5 !important;
    color: #0B0B0B !important;
}

.rc-slider-tooltip-arrow {
    border-top-color: #F5F5F5 !important;
}


/* ============================================================
   INPUTS
   ============================================================ */

input[type="number"],
input[type="text"] {
    background-color: #0B0B0B !important;
    color: #F5F5F5 !important;
    border: 1px solid #333333 !important;
}

input[type="number"]:focus,
input[type="text"]:focus {
    border-color: #F5F5F5 !important;
}


/* ============================================================
   CHECKBOXES
   ============================================================ */

input[type="checkbox"] {
    accent-color: #F5F5F5;
}

label {
    color: #F5F5F5;
}


/* ============================================================
   GRAPH
   ============================================================ */

.js-plotly-plot {
    border-radius: 10px;
    overflow: hidden;
}


/* ============================================================
   DATA TABLE
   ============================================================ */

.dash-table-container .dash-spreadsheet-container {
    background: #0B0B0B !important;
}

.dash-table-container .dash-spreadsheet-inner table {
    border-collapse: collapse !important;
}

.dash-table-container .dash-spreadsheet-inner td {
    background-color: #0B0B0B !important;
    color: #F5F5F5 !important;
    border: 1px solid #333333 !important;
}

.dash-table-container .dash-spreadsheet-inner th {
    background-color: #1F1F1F !important;
    color: #F5F5F5 !important;
    border: 1px solid #333333 !important;
}

.dash-table-container .dash-spreadsheet-inner tr:hover td {
    background-color: #1F1F1F !important;
    color: #F5F5F5 !important;
}


/* ============================================================
   GENERAL
   ============================================================ */

._dash-loading {
    color: #F5F5F5 !important;
}

.dash-tooltip {
    background-color: #F5F5F5 !important;
    color: #0B0B0B !important;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 1300px) {
    .layout {
        grid-template-columns: 1fr;
    }

    .cards {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .cards {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .page {
        padding: 10px;
    }
}
"""
# ============================================================================
# Dash app
# ============================================================================

app = Dash(
    __name__,
    title="Motion Analysis Dashboard",
)

server = app.server


app.layout = html.Div(
    className="page",
    children=[
        # ------------------------------------------------------------
        # Persistent stores
        # ------------------------------------------------------------
        dcc.Store(
            id="csv-uploaded-path",
        ),
        dcc.Store(
            id="video-uploaded-path",
        ),
        dcc.Store(
            id="data-meta",
        ),
        dcc.Store(
            id="csv-data-store",
        ),
        dcc.Store(
            id="video-data-store",
        ),

        # ------------------------------------------------------------
        # Header
        # ------------------------------------------------------------
        html.Div(
            className="header",
            children=[
                html.H1(
                    "CSV + Video Motion Verification Dashboard"
                ),
                html.P(
                    "Click a graph point to jump to the corresponding "
                    "video frame. All dashboard values follow that frame."
                ),
            ],
        ),

        html.Div(
            className="layout",
            children=[
                # ====================================================
                # LEFT
                # ====================================================
                html.Div(
                    className="panel",
                    children=[
                        html.Div(
                            "1. CSV",
                            className="section-title",
                        ),

                        dcc.Upload(
                            id="csv-upload",
                            children=html.Div(
                                [
                                    html.Button(
                                        "Choose CSV file"
                                    ),
                                    html.Div(
                                        "or drag and drop a CSV here",
                                        style={
                                            "marginTop": "8px",
                                            "color": "#94a3b8",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ]
                            ),
                            multiple=False,
                            accept="",
                            max_size=-1,
                            className="upload-box",
                        ),

                        html.Div(
                            id="csv-file-name",
                            className="file-name",
                            children="No CSV uploaded",
                        ),

                        html.Div(
                            "Direct CSV path (recommended for local files)",
                            className="control-title",
                        ),

                        dcc.Input(
                            id="csv-path",
                            type="text",
                            placeholder=(
                                r"C:\Users\...\metrics.csv"
                            ),
                            className="path-input",
                        ),

                        html.Div(
                            "Use an absolute Windows path when the file is "
                            "large or browser upload is inconvenient.",
                            className="path-help",
                        ),

                        html.Div(
                            "2. Matching video",
                            className="section-title",
                            style={
                                "marginTop": "22px"
                            },
                        ),

                        dcc.Upload(
                            id="video-upload",
                            children=html.Div(
                                [
                                    html.Button(
                                        "Choose video file"
                                    ),
                                    html.Div(
                                        "or drag and drop a video here",
                                        style={
                                            "marginTop": "8px",
                                            "color": "#94a3b8",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ]
                            ),
                            multiple=False,
                            # Deliberately no MIME accept filter.
                            # Browser MIME detection is inconsistent.
                            accept="",
                            max_size=-1,
                            className="upload-box",
                        ),

                        html.Div(
                            id="video-file-name",
                            className="file-name",
                            children="No video uploaded",
                        ),

                        html.Div(
                            "Direct video path (recommended for large videos)",
                            className="control-title",
                        ),

                        dcc.Input(
                            id="video-path",
                            type="text",
                            placeholder=(
                                r"C:\Users\...\video.mp4"
                            ),
                            className="path-input",
                        ),

                        html.Div(
                            "Supported formats depend on the OpenCV "
                            "installation: MP4, AVI, MOV, MKV, etc.",
                            className="path-help",
                        ),

                        html.Div(
                            "3. Metrics",
                            className="section-title",
                            style={
                                "marginTop": "22px"
                            },
                        ),

                        dcc.Dropdown(
                            id="metric-selector",
                            options=[
                                {
                                    "label": metric,
                                    "value": metric,
                                }
                                for metric in METRICS
                            ],
                            value=METRICS[:3],
                            multi=True,
                            clearable=True,
                            placeholder=(
                                "Select metric(s) to plot"
                            ),
                        ),

                        html.Div(
                            "Smoothing window",
                            className="control-title",
                        ),

                        dcc.Slider(
                            id="smoothing",
                            min=1,
                            max=61,
                            step=2,
                            value=5,
                            marks={
                                1: "1",
                                15: "15",
                                31: "31",
                                61: "61",
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),

                        dcc.Checklist(
                            id="plot-options",
                            options=[
                                {
                                    "label": "Show raw",
                                    "value": "raw",
                                },
                                {
                                    "label": "Show smoothed",
                                    "value": "smooth",
                                },
                                {
                                    "label": "Show markers",
                                    "value": "markers",
                                },
                            ],
                            value=[
                                "raw",
                                "smooth",
                            ],
                            style={
                                "display": "grid",
                                "gap": "7px",
                                "marginTop": "16px",
                            },
                        ),

                        html.Div(
                            "Line width",
                            className="control-title",
                        ),

                        dcc.Slider(
                            id="line-width",
                            min=1,
                            max=6,
                            step=0.5,
                            value=2,
                        ),

                        html.Div(
                            "Plot height",
                            className="control-title",
                        ),

                        dcc.Slider(
                            id="plot-height",
                            min=400,
                            max=900,
                            step=50,
                            value=620,
                            marks={
                                400: "400",
                                650: "650",
                                900: "900",
                            },
                        ),

                        html.Div(
                            "CSV frame offset",
                            className="control-title",
                        ),

                        dcc.Input(
                            id="frame-offset",
                            type="number",
                            value=0,
                            step=1,
                            className="path-input",
                        ),

                        html.Div(
                            "video frame = CSV frame + offset",
                            className="path-help",
                        ),

                        html.Div(
                            "Visible frame start",
                            className="control-title",
                        ),

                        dcc.Input(
                            id="x-start",
                            type="number",
                            placeholder="Dataset start",
                            className="path-input",
                        ),

                        html.Div(
                            "Visible frame end",
                            className="control-title",
                        ),

                        dcc.Input(
                            id="x-end",
                            type="number",
                            placeholder="Dataset end",
                            className="path-input",
                        ),

                        html.Div(
                            id="status",
                            className="status",
                            children=(
                                "Upload files or enter their "
                                "local paths."
                            ),
                        ),
                    ],
                ),

                # ====================================================
                # RIGHT
                # ====================================================
                html.Div(
                    className="panel",
                    children=[
                        html.Div(
                            id="cards",
                            className="cards",
                            children=[
                                make_card(
                                    "Current frame",
                                    "—",
                                ),
                                make_card(
                                    "Video frame",
                                    "—",
                                ),
                                make_card(
                                    "Metrics selected",
                                    "0",
                                ),
                                make_card(
                                    "Frame average",
                                    "—",
                                ),
                                make_card(
                                    "Frame spread",
                                    "—",
                                ),
                                make_card(
                                    "Frame max",
                                    "—",
                                ),
                            ],
                        ),

                        html.Div(
                            "Current CSV frame",
                            className="frame-label",
                        ),

                        dcc.Slider(
                            id="frame-slider",
                            min=0,
                            max=1,
                            value=0,
                            step=1,
                            updatemode="mouseup",
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),

                        html.Div(
                            "Video frame",
                            className="section-title",
                            style={
                                "marginTop": "18px"
                            },
                        ),

                        html.Div(
                            id="frame-viewer",
                            className="video-box",
                            children=html.Div(
                                "Upload a video or enter its local path.",
                                className="frame-placeholder",
                            ),
                        ),

                        html.Div(
                            "Interactive graph",
                            className="section-title",
                            style={
                                "marginTop": "18px"
                            },
                        ),

                        dcc.Graph(
                            id="motion-graph",
                            figure=empty_figure(),
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "doubleClick": "reset",
                            },
                        ),

                        html.Div(
                            "Selected-column statistics",
                            className="section-title",
                            style={
                                "marginTop": "18px"
                            },
                        ),

                        dash_table.DataTable(
                            id="stats-table",
                            columns=[
                                {
                                    "name": col,
                                    "id": col,
                                }
                                for col in [
                                    "Metric",
                                    "Current",
                                    "Mean",
                                    "Median",
                                    "Std",
                                    "Min",
                                    "Max",
                                    "Range",
                                    "Valid N",
                                ]
                            ],
                            data=[],
                            style_table={
                                "overflowX": "auto",
                            },
                            style_header={
                                "backgroundColor": "#1e293b",
                                "color": "#f8fafc",
                                "fontWeight": "700",
                            },
                            style_cell={
                                "backgroundColor": "#0f172a",
                                "color": "#cbd5e1",
                                "padding": "10px",
                                "border": (
                                    "1px solid #263244"
                                ),
                                "textAlign": "left",
                                "minWidth": "90px",
                            },
                            style_data_conditional=[
                                {
                                    "if": {
                                        "column_id": "Current"
                                    },
                                    "fontWeight": "700",
                                    "color": "#f8fafc",
                                }
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================================
# Upload callback
# ============================================================================

@app.callback(
    Output(
        "csv-uploaded-path",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "video-uploaded-path",
        "data",
        allow_duplicate=True,
    ),
    Output("csv-file-name", "children"),
    Output("video-file-name", "children"),
    Input(
        "csv-upload",
        "contents",
    ),
    Input(
        "video-upload",
        "contents",
    ),
    State(
        "csv-upload",
        "filename",
    ),
    State(
        "video-upload",
        "filename",
    ),
    prevent_initial_call=True,
)
def process_uploads(
    csv_contents,
    video_contents,
    csv_filename,
    video_filename,
):
    # The stores are updated independently where possible.
    csv_path = dash.no_update
    video_path = dash.no_update

    csv_label = dash.no_update
    video_label = dash.no_update

    errors = []

    if csv_contents:
        try:
            csv_path = save_upload(
                csv_contents,
                csv_filename,
                "csv",
            )
            csv_label = (
                f"Uploaded CSV: "
                f"{csv_filename}"
            )
        except Exception as exc:
            csv_path = None
            csv_label = (
                f"CSV upload failed: {exc}"
            )
            errors.append(
                f"CSV upload failed: {exc}"
            )

    if video_contents:
        try:
            video_path = save_upload(
                video_contents,
                video_filename,
                "video",
            )
            video_label = (
                f"Uploaded video: "
                f"{video_filename}"
            )
        except Exception as exc:
            video_path = None
            video_label = (
                f"Video upload failed: {exc}"
            )
            errors.append(
                f"Video upload failed: {exc}"
            )

    # Keep callback valid even if only one upload changed.
    return (
        csv_path,
        video_path,
        csv_label,
        video_label,
    )


# ============================================================================
# Main dashboard callback
# ============================================================================

@app.callback(
    Output("csv-data-store", "data"),
    Output("video-data-store", "data"),
    Output("data-meta", "data"),
    Output("status", "children"),
    Output("frame-slider", "min"),
    Output("frame-slider", "max"),
    Output("frame-slider", "value"),
    Input("csv-path", "value"),
    Input("video-path", "value"),
    Input("csv-uploaded-path", "data"),
    Input("video-uploaded-path", "data"),
)
def load_files(
    csv_path_text,
    video_path_text,
    uploaded_csv,
    uploaded_video,
):
    csv_path = resolve_path(
        uploaded_csv,
        csv_path_text,
    )

    video_path = resolve_path(
        uploaded_video,
        video_path_text,
    )

    if not csv_path:
        return (
            None,
            None,
            None,
            "CSV not found. Upload one or enter its local path.",
            0,
            1,
            0,
        )

    try:
        df = read_csv(csv_path)
    except Exception as exc:
        return (
            None,
            None,
            None,
            f"CSV error:\n{exc}",
            0,
            1,
            0,
        )

    video_info = get_video_info(
        video_path
    )

    status_lines = [
        (
            f"CSV: {Path(csv_path).name}"
            f" | {len(df):,} rows"
            f" | frames "
            f"{fmt(df['frame'].min(), 0)}–"
            f"{fmt(df['frame'].max(), 0)}"
        )
    ]

    if video_path:
        if video_info["frames"] > 0:
            status_lines.append(
                (
                    f"Video: "
                    f"{Path(video_path).name}"
                    f" | {video_info['frames']:,} frames"
                    + (
                        f" | "
                        f"{video_info['fps']:.2f} FPS"
                        if video_info["fps"]
                        else ""
                    )
                    + (
                        f" | "
                        f"{video_info['width']}"
                        f"×"
                        f"{video_info['height']}"
                        if video_info["width"]
                        and video_info["height"]
                        else ""
                    )
                )
            )
        else:
            status_lines.append(
                "Video was found, but OpenCV could not "
                "read its metadata."
            )
    else:
        status_lines.append(
            "Video: not provided yet."
        )

    status_lines.append(
        "Interaction: click a graph point to jump "
        "to that frame."
    )

    frame_min = int(
        round(
            float(
                df["frame"].min()
            )
        )
    )

    frame_max = int(
        round(
            float(
                df["frame"].max()
            )
        )
    )

    initial_frame = frame_min

    # Store CSV as JSON rather than retaining a browser-side Python object.
    csv_json = df.to_json(
        orient="split"
    )

    meta = {
        "csv_path": csv_path,
        "video_path": video_path,
        "frame_min": frame_min,
        "frame_max": frame_max,
    }

    video_data = {
        "path": video_path
    } if video_path else None

    return (
        csv_json,
        video_data,
        meta,
        "\n".join(status_lines),
        frame_min,
        frame_max,
        initial_frame,
    )


# ============================================================================
# Dashboard output callback
# ============================================================================

@app.callback(
    Output("motion-graph", "figure"),
    Output("frame-viewer", "children"),
    Output("stats-table", "data"),
    Output("cards", "children"),
    Input("frame-slider", "value"),
    Input("metric-selector", "value"),
    Input("smoothing", "value"),
    Input("plot-options", "value"),
    Input("line-width", "value"),
    Input("plot-height", "value"),
    Input("frame-offset", "value"),
    Input("x-start", "value"),
    Input("x-end", "value"),
    State("csv-data-store", "data"),
    State("video-data-store", "data"),
)
def update_dashboard(
    slider_frame,
    selected,
    smoothing,
    plot_options,
    line_width,
    plot_height,
    frame_offset,
    x_start,
    x_end,
    csv_json,
    video_data,
):
    if not csv_json:
        return (
            empty_figure(),
            html.Div(
                "Upload CSV and video, or enter their local paths.",
                className="frame-placeholder",
            ),
            [],
            [
                make_card(
                    "Current frame",
                    "—",
                ),
                make_card(
                    "Video frame",
                    "—",
                ),
                make_card(
                    "Metrics selected",
                    "0",
                ),
                make_card(
                    "Frame average",
                    "—",
                ),
                make_card(
                    "Frame spread",
                    "—",
                ),
                make_card(
                    "Frame max",
                    "—",
                ),
            ],
        )

    try:
        df = pd.read_json(
            csv_json,
            orient="split",
        )
    except Exception as exc:
        return (
            empty_figure(
                f"Could not restore CSV: {exc}"
            ),
            html.Div(
                str(exc),
                className="frame-placeholder",
            ),
            [],
            [],
        )

    if df.empty:
        return (
            empty_figure(
                "CSV contains no rows"
            ),
            html.Div(
                "CSV contains no usable rows.",
                className="frame-placeholder",
            ),
            [],
            [],
        )

    selected = selected or []

    if slider_frame is None:
        slider_frame = float(
            df["frame"].iloc[0]
        )

    actual_frame = nearest_csv_frame(
        df,
        float(slider_frame),
    )

    fig = make_figure(
        df=df,
        selected=selected,
        smoothing=int(
            smoothing or 1
        ),
        plot_options=plot_options or [],
        line_width=float(
            line_width or 2
        ),
        current_frame=actual_frame,
        x_start=x_start,
        x_end=x_end,
        plot_height=int(
            plot_height or 620
        ),
    )

    video_path = None

    if video_data:
        video_path = video_data.get(
            "path"
        )

    image, video_frame_number = (
        read_video_frame(
            video_path,
            actual_frame,
            int(
                frame_offset or 0
            ),
        )
    )

    data_uri = image_to_data_uri(
        image
    )

    if data_uri:
        viewer = html.Div(
            [
                html.Img(
                    src=data_uri,
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "600px",
                    },
                ),
                html.Div(
                    (
                        f"CSV frame: "
                        f"{fmt(actual_frame, 0)}"
                        " | "
                        f"Video frame: "
                        f"{video_frame_number}"
                    ),
                    style={
                        "padding": "8px 12px",
                        "color": "#94a3b8",
                        "fontSize": "12px",
                        "width": "100%",
                    },
                ),
            ],
            style={
                "width": "100%",
                "textAlign": "center",
            },
        )
    else:
        if video_path:
            viewer_message = (
                "Unable to decode this video frame."
            )
        else:
            viewer_message = (
                "No video loaded."
            )

        viewer = html.Div(
            [
                html.Div(
                    viewer_message,
                    className="frame-placeholder",
                ),
                html.Div(
                    f"Selected CSV frame: "
                    f"{fmt(actual_frame, 0)}",
                    style={
                        "color": "#64748b",
                        "fontSize": "12px",
                    },
                ),
            ]
        )

    statistics = make_statistics(
        df,
        selected,
        actual_frame,
    )

    cards = make_cards(
        df,
        selected,
        actual_frame,
        video_frame_number,
        int(
            frame_offset or 0
        ),
    )

    return (
        fig,
        viewer,
        statistics,
        cards,
    )


# ============================================================================
# Graph click -> selected frame
# ============================================================================

@app.callback(
    Output(
        "frame-slider",
        "value",
        allow_duplicate=True,
    ),
    Input(
        "motion-graph",
        "clickData",
    ),
    State(
        "csv-data-store",
        "data",
    ),
    prevent_initial_call=True,
)
def graph_to_frame(
    click_data,
    csv_json,
):
    if (
        not click_data
        or not csv_json
    ):
        return dash.no_update

    points = click_data.get(
        "points",
        [],
    )

    if not points:
        return dash.no_update

    x_value = points[0].get(
        "x"
    )

    if x_value is None:
        return dash.no_update

    try:
        df = pd.read_json(
            csv_json,
            orient="split",
        )

        return nearest_csv_frame(
            df,
            float(x_value),
        )
    except Exception:
        return dash.no_update


# ============================================================================
# Reset frame range when CSV changes
# ============================================================================
#
# The main loading callback already resets the slider, so no additional
# callback is necessary.
# ============================================================================


# ============================================================================
# HTML shell
# ============================================================================

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
        {CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""


# ============================================================================
# Start
# ============================================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )
