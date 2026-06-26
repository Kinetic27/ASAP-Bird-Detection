# src/core/loader.py

import cv2
import queue
import threading
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from src.utils.config import PATCH_SIZE, MIN_OVERLAP
from src.utils.tiling import build_patch_offsets


class ThreadedStreamLoader:
    def __init__(
        self,
        source,
        patch_size=PATCH_SIZE,
        min_overlap=MIN_OVERLAP,
        target_size=None,
        global_size=None,
        queue_size=32,
        crop_mode=False,
        stride=1,
        offset=0,
        drop_if_full=True,
        source_fps_cap=None,
        shared_buffer=None,
        global_shared_buffer=None,
    ):
        # Use FFMPEG for better RTSP/Stream support
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.target_size = target_size
        if self.target_size is not None:
            self.w, self.h = self.target_size

        self.crop_mode = crop_mode
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.stride = stride
        self.offset = offset
        self.drop_if_full = drop_if_full
        self.source_fps_cap = float(source_fps_cap) if source_fps_cap else None
        if self.source_fps_cap is not None and self.source_fps_cap <= 0:
            self.source_fps_cap = None

        # Ring Buffer Support
        self.shared_buffer = shared_buffer
        self.buffer_size = 0
        self.buffer_idx = 0
        if self.shared_buffer is not None:
            self.buffer_size = self.shared_buffer.shape[0]
        # Global Buffer Support
        self.global_shared_buffer = global_shared_buffer
        self.global_size = global_size
        if self.global_shared_buffer is not None and self.global_size is None:
            self.global_size = self.patch_size  # Default fallback

        # Pre-calc patch offsets (Fast indexing)
        self.offsets = build_patch_offsets(
            self.w, self.h, self.patch_size, self.min_overlap
        )

        self.q = queue.Queue(maxsize=queue_size)
        self.raw_q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.reader_finished = False
        self.pending_tasks = 0
        self.pending_tasks_lock = threading.Lock()

        # Profiling stats
        self.time_read = 0.0
        self.time_preprocess = 0.0
        self.frame_count = 0

        self.pool = ThreadPoolExecutor(max_workers=4)

    def start(self):
        self.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.preprocessor_thread = threading.Thread(
            target=self.preprocessor_loop, daemon=True
        )
        self.reader_thread.start()
        self.preprocessor_thread.start()
        return self

    def reader_loop(self):
        current_idx = 0
        next_read_due = time.time()
        while not self.stopped:
            if not self.raw_q.full():
                if self.source_fps_cap is not None:
                    now = time.time()
                    if now < next_read_due:
                        time.sleep(next_read_due - now)
                    next_read_due = max(time.time(), next_read_due) + (
                        1.0 / self.source_fps_cap
                    )
                t0 = time.time()
                if self.stride > 1 and (current_idx % self.stride != self.offset):
                    ret = self.cap.grab()
                    frame = None
                else:
                    ret, frame = self.cap.read()

                t1 = time.time()
                self.time_read += t1 - t0

                if not ret:
                    self.reader_finished = True
                    break

                if frame is not None:
                    self.raw_q.put((current_idx, frame))
                current_idx += 1
            else:
                time.sleep(0.001)
        self.reader_finished = True
        self.cap.release()

    def preprocessor_loop(self):
        while not self.stopped:
            if self.reader_finished and self.raw_q.empty():
                break
            try:
                item = self.raw_q.get(timeout=0.1)
                idx, frame = item
            except queue.Empty:
                continue

            def process_and_queue(idx, frame, offsets):
                try:
                    if self.stopped:
                        return

                    t1 = time.time()
                    self.frame_count += 1
                    h, w = frame.shape[:2]
                    if self.target_size:
                        tw, th = self.target_size
                        if (w, h) != (tw, th):
                            if self.crop_mode:
                                cx, cy = w // 2, h // 2
                                x1, y1 = max(0, cx - tw // 2), max(0, cy - th // 2)
                                frame = frame[y1 : y1 + th, x1 : x1 + tw]
                            else:
                                frame = cv2.resize(
                                    frame,
                                    self.target_size,
                                    interpolation=cv2.INTER_LINEAR,
                                )

                    # Ring Buffer Optimization (Zero Copy Logic)
                    if self.shared_buffer is not None:
                        slot_idx = idx % self.buffer_size
                        self.shared_buffer[slot_idx] = torch.from_numpy(
                            np.ascontiguousarray(frame)
                        )

                        if self.global_shared_buffer is not None:
                            g_size = self.global_size
                            global_img = cv2.resize(
                                frame, (g_size, g_size), interpolation=cv2.INTER_LINEAR
                            )
                            self.global_shared_buffer[slot_idx] = torch.from_numpy(
                                np.ascontiguousarray(global_img)
                            )

                        processed_result = int(slot_idx)
                    else:
                        tensor = torch.from_numpy(np.ascontiguousarray(frame))
                        processed_result = tensor.share_memory_()

                    t_proc = time.time() - t1
                    # Note: this is slightly imprecise metrics for multi-threading but gives an idea
                    self.time_preprocess += t_proc

                    if self.stopped:
                        return

                    while not self.stopped:
                        try:
                            self.q.put(
                                (idx, processed_result, offsets), timeout=0.05
                            )
                            break
                        except queue.Full:
                            continue
                except Exception as e:
                    if not self.stopped:
                        print(f"[Loader] Error processing frame {idx}: {e}")
                finally:
                    with self.pending_tasks_lock:
                        self.pending_tasks -= 1

            # Use thread pool to process and queue directly
            with self.pending_tasks_lock:
                self.pending_tasks += 1
            self.pool.submit(process_and_queue, idx, frame, self.offsets)

    def more(self):
        with self.pending_tasks_lock:
            pending_tasks = self.pending_tasks
        return (
            not self.reader_finished
            or not self.raw_q.empty()
            or pending_tasks > 0
            or not self.q.empty()
        )

    def stop(self):
        self.stopped = True
        self.reader_finished = True
        while not self.raw_q.empty():
            try:
                self.raw_q.get_nowait()
            except queue.Empty:
                break
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.reader_thread.join(timeout=1)
        self.preprocessor_thread.join(timeout=1)
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self.pool.shutdown(wait=True)
        if self.frame_count > 0:
            print(
                f"[Loader Profile] Read: {self.time_read:.2f}s, Preprocess: {self.time_preprocess:.2f}s over {self.frame_count} frames"
            )
