from __future__ import annotations

import os


VALID_DROP_POLICIES = {"drop_oldest", "drop_newest", "latest_only"}


def is_stream_source(input_path) -> bool:
    value = str(input_path)
    return value.startswith(("rtsp", "rtmp", "http", "tcp")) or value.isdigit()


def should_pause_dispatch(*, in_flight: int, max_in_flight: int) -> bool:
    """Return True when dispatch should yield to result collection.

    Dispatch must pause once the backlog reaches the cap. This keeps file
    replay from draining to EOF before workers can publish results.
    """
    return int(in_flight) >= int(max_in_flight)


def normalize_queue_controls(
    max_in_flight,
    drop_policy,
    input_fps_cap,
    *,
    default_max_in_flight: int = 64,
):
    if max_in_flight is None or max_in_flight < 1:
        max_in_flight = default_max_in_flight

    drop_policy = str(drop_policy).strip().lower()
    if drop_policy not in VALID_DROP_POLICIES:
        drop_policy = "drop_oldest"
    if drop_policy == "latest_only":
        max_in_flight = 1

    if input_fps_cap is not None:
        try:
            input_fps_cap = float(input_fps_cap)
            if input_fps_cap <= 0:
                input_fps_cap = None
        except Exception:
            input_fps_cap = None

    return max_in_flight, drop_policy, input_fps_cap


def build_video_output_paths(input_path, output_path: str, *, is_stream: bool, save: bool, save_json: bool):
    filename = os.path.basename(str(input_path)) if not is_stream else "stream_output.mp4"
    save_path = os.path.join(output_path, filename) if save else None
    json_save_path = (
        os.path.join(output_path, os.path.splitext(filename)[0] + ".json")
        if save_json
        else None
    )
    return filename, save_path, json_save_path


def serialize_frame_detections(frame_idx: int, boxes) -> dict:
    frame_res = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            frame_res.append(
                {
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "conf": float(box[4]),
                    "class": int(box[5]),
                }
            )
    return {"frame_idx": int(frame_idx), "detections": frame_res}
