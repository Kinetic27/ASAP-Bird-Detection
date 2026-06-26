import os
import sys
import argparse
import time
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory
from ultralytics import YOLO
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.tiling import build_patch_windows
from src.core.worker_postprocess import (
    append_boxes,
    apply_nms,
    format_worker_output,
    offset_patch_boxes,
    scale_global_boxes,
)
from src.core.shared_memory_utils import untrack_shared_memory

# Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def parse_classes(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _predict_batch(model, batch_slice, args, classes, device_id, patch_size):
    return model(
        batch_slice,
        imgsz=patch_size,
        conf=args.conf,
        iou=args.iou_thres,
        classes=classes,
        augment=args.augment,
        verbose=False,
        device=device_id if device_id == "cpu" else int(device_id),
        save=False,
    )


def _load_model(args, classes, device_id, patch_size, *, warmup=True):
    if device_id != "cpu" and args.worker_slot > 0:
        time.sleep(min(args.worker_slot * 0.25, 2.0))

    sys.stderr.write(f"[Worker {device_id}:{args.worker_slot}] Loading {args.model}...\n")
    model = YOLO(args.model)

    if warmup:
        warmup_frame = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        _predict_batch(
            model,
            [warmup_frame] * args.batch_size,
            args,
            classes,
            device_id,
            patch_size,
        )
        if device_id != "cpu":
            torch.cuda.synchronize()

    sys.stderr.write(f"[Worker {device_id}:{args.worker_slot}] Ready.\n")
    return model


def _is_tensorrt_context_error(exc: Exception) -> bool:
    message = str(exc)
    return "set_input_shape" in message and "NoneType" in message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--shm-name", type=str, required=True)
    parser.add_argument("--shm-shape", type=str, required=True)  # "N,H,W,C"
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--min-overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--global-shm-name", type=str, default=None)
    parser.add_argument("--global-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--worker-slot", type=int, default=0)
    args = parser.parse_args()
    classes = parse_classes(args.classes)

    # 1. Setup Device
    device_id = args.device
    if device_id != "cpu":
        torch.cuda.set_device(int(device_id))

    # 2. Attach SHM
    try:
        shm = SharedMemory(name=args.shm_name)
        untrack_shared_memory(shm)
        shape = tuple(map(int, args.shm_shape.split(",")))
        # Create Numpy View
        buffer_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        sys.stderr.write(f"SHM Error: {e}\n")
        sys.exit(1)

    # 2b. Attach Global SHM (Optional)
    global_buffer_array = None
    if args.global_context and args.global_shm_name:
        try:
            g_shm = SharedMemory(name=args.global_shm_name)
            untrack_shared_memory(g_shm)
            g_shape = (shape[0], args.global_size, args.global_size, 3)
            global_buffer_array = np.ndarray(g_shape, dtype=np.uint8, buffer=g_shm.buf)
        except Exception as e:
            sys.stderr.write(f"Global SHM Error: {e}\n")

    # 3. Load Model and force TensorRT context creation before accepting work.
    try:
        model = _load_model(args, classes, device_id, patch_size=args.patch_size)
    except Exception as e:
        sys.stderr.write(f"[Worker {device_id}:{args.worker_slot}] Model Error: {e}\n")
        sys.exit(1)

    # 4. Loop
    stdin = sys.stdin
    stdout = sys.stdout

    # --- Pre-calculate Tiling ---
    last_shape = None
    cached_tiling = None
    patch_size = args.patch_size

    while True:
        line = stdin.readline()
        if not line:
            break

        try:
            t_start_total = time.time()
            cmd = line.strip()
            frame_idx = int(cmd)
            # --- Modulo Access for Ring Buffer ---
            buffer_idx = frame_idx % shape[0]

            # Read Frame from SHM (Direct view, zero copy)
            # Safe because ring buffer is 64 slots and frame is used only for inference
            frame = np.array(buffer_array[buffer_idx], copy=False)

            # --- Tiling Logic (Cached) ---
            if cached_tiling is None or frame.shape[:2] != last_shape:
                h, w = frame.shape[:2]
                last_shape = (h, w)
                cached_tiling = build_patch_windows(
                    w, h, args.patch_size, args.min_overlap
                )

            patches = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in cached_tiling]
            # --- Global Context (Optimized: Pull from SHM) ---
            if args.global_context:
                if global_buffer_array is not None:
                    # Pull pre-resized frame from loader
                    global_frame = np.array(global_buffer_array[buffer_idx], copy=False)
                else:
                    # Fallback if SHM not provided (e.g. legacy/direct use)
                    global_frame = cv2.resize(frame, (args.global_size, args.global_size))
                patches.append(global_frame)
                # coords_only for global frame is special, we handle it in post-process

            # --- Measure Inference Time ---
            t_start_inf = time.time()

            # --- Batch Inference ---
            results = []
            batch_size = args.batch_size  # Use Arg

            # Allocate padding only when the final batch is short. The paper
            # 1280px path has exactly 8 patches with batch_size=8, so padding is
            # usually unnecessary; avoid zero-filling a 1280x1280 image per frame.
            pad_img = None

            for i in range(0, len(patches), batch_size):
                batch_slice = patches[i : i + batch_size]
                current_len = len(batch_slice)

                # Padding for TRT fixed batch
                if current_len < batch_size:
                    if pad_img is None:
                        pad_img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    batch_slice = batch_slice + [pad_img] * (batch_size - current_len)

                # Inference. A TensorRT execution context can occasionally be
                # invalidated in multi-process runs; reload once and retry the
                # batch instead of silently returning empty detections.
                try:
                    batch_res = _predict_batch(model, batch_slice, args, classes, device_id, patch_size)
                except AttributeError as e:
                    if not _is_tensorrt_context_error(e):
                        raise
                    sys.stderr.write(
                        f"[Worker {device_id}:{args.worker_slot}] TensorRT context invalid; "
                        "reloading model and retrying once.\n"
                    )
                    model = _load_model(args, classes, device_id, patch_size, warmup=False)
                    batch_res = _predict_batch(model, batch_slice, args, classes, device_id, patch_size)
                results.extend(batch_res[:current_len])

            t_end_inf = time.time()
            inf_duration = t_end_inf - t_start_inf

            # --- Optimized Post Process ---
            final_boxes_list = []

            for i, r in enumerate(results):
                # Move ALL boxes for this patch to CPU at once
                # b.data shape is [N, 6] where 6 is [x1, y1, x2, y2, conf, cls]
                boxes_data = r.boxes.data.cpu().numpy()
                if boxes_data.shape[0] == 0:
                    continue

                # Check if this is the Global Frame
                is_global = args.global_context and (i == len(results) - 1)

                if is_global:
                    scale_global_boxes(boxes_data, width=w, height=h, global_size=args.global_size)
                else:
                    offset_x, offset_y = cached_tiling[i][0], cached_tiling[i][1]
                    offset_patch_boxes(boxes_data, offset_x=offset_x, offset_y=offset_y)

                append_boxes(final_boxes_list, boxes_data)

            # --- NMS on Combined Boxes ---
            t_start_nms = time.time()
            final_boxes = apply_nms(
                final_boxes_list,
                score_threshold=args.conf,
                nms_threshold=args.iou_thres,
            )
            t_end_nms = time.time()
            nms_duration = t_end_nms - t_start_nms

            t_end_total = time.time()
            total_duration = t_end_total - t_start_total

            out_str = format_worker_output(
                frame_idx,
                total_duration,
                inf_duration,
                nms_duration,
                final_boxes,
            )
            stdout.write(out_str + "\n")
            stdout.flush()

        except ValueError:
            continue
        except Exception as e:
            failed_frame = locals().get("frame_idx", "unknown")
            sys.stderr.write(
                f"[Worker {device_id}:{args.worker_slot}] Error while processing frame "
                f"{failed_frame}: {type(e).__name__}: {e}\n"
            )
            sys.stderr.flush()
            sys.exit(1)


if __name__ == "__main__":
    main()
