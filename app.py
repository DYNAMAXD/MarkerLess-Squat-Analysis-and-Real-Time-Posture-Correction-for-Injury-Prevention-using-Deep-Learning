"""
app.py — HF Space UI.

Since there's no terminal access in the Space, this UI includes a live
"Debug Console" textbox that streams every log line from keypoint_pipeline
(and any uncaught traceback) in real time, so you can see stderr-equivalent
output without SSH/logs access.
"""

import logging
import queue
import traceback
import spaces
import gradio as gr

from keypoint_pipeline import process_video, LOGGER


# --------------------------------------------------------------------------
# Debug console: a logging.Handler that pushes lines into a thread-safe
# queue, drained by the Gradio generator to update the console live.
# --------------------------------------------------------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


@spaces.GPU(duration=1)  # satisfies ZeroGPU's startup check; we force CPU inside anyway
def _run(video_file, use_3d, det_conf):
    return process_video(video_file, use_3d=use_3d, det_conf=det_conf)

def run_pipeline(video_file, use_3d, det_conf):
    log_queue: queue.Queue = queue.Queue()
    handler = QueueLogHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)

    console_text = ""
    result_csv = None
    result_video = None

    def flush_queue():
        nonlocal console_text
        drained = False
        while not log_queue.empty():
            console_text += log_queue.get() + "\n"
            drained = True
        return drained

    if video_file is None:
        console_text += "ERROR: no video uploaded.\n"
        yield console_text, None, None
        LOGGER.removeHandler(handler)
        return

    console_text += f"Starting pipeline on: {video_file}\n"
    yield console_text, None, None

    import threading

    result_holder = {}

    def worker():
        try:
            csv_path, overlay_path = process_video(video_file, use_3d=use_3d, det_conf=det_conf)
            result_holder["csv"] = csv_path
            result_holder["video"] = overlay_path
        except Exception:
            result_holder["error"] = traceback.format_exc()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while t.is_alive():
        if flush_queue():
            yield console_text, None, None
        t.join(timeout=0.3)

    flush_queue()  # final drain

    if "error" in result_holder:
        console_text += "\n=== PIPELINE FAILED ===\n" + result_holder["error"] + "\n"
        yield console_text, None, None
    else:
        result_csv = result_holder.get("csv")
        result_video = result_holder.get("video")
        console_text += "\n=== DONE ===\n"
        yield console_text, result_csv, result_video

    LOGGER.removeHandler(handler)


with gr.Blocks(title="AlphaPose + MotionBERT Keypoint Extraction") as demo:
    gr.Markdown(
        "## Video -> 2D Keypoints (AlphaPose halpe26) -> 3D Lift (MotionBERT)\n"
        "Upload a video. Output: CSV of joint positions per frame + an overlay video "
        "with 2D keypoints drawn on it. Mesh generation will be added later."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Input video", format="mp4")
            use_3d = gr.Checkbox(value=True, label="Also run MotionBERT 3D lift")
            det_conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="YOLOv11 person detection confidence")
            run_btn = gr.Button("Run pipeline", variant="primary")

        with gr.Column(scale=1):
            csv_out = gr.File(label="Keypoints CSV")
            video_out = gr.Video(label="Overlay video")

    gr.Markdown("### Debug Console (live stderr/stdout equivalent — no terminal needed)")
    debug_console = gr.Textbox(
        label="Debug output",
        lines=20,
        max_lines=30,
        interactive=False,
        autoscroll=True,
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_in, use_3d, det_conf],
        outputs=[debug_console, csv_out, video_out],
    )

if __name__ == "__main__":
    demo.queue().launch(show_api=False)
