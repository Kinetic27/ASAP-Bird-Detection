# src/core/inference.py

import subprocess
import sys
import threading
import queue
import json
import time
import os
from multiprocessing.shared_memory import SharedMemory
import cv2
import torch
import numpy as np
from tqdm import tqdm

from src.utils.config import (
    PATCH_SIZE,
    MIN_OVERLAP,
    CONF_THRES,
    DEFAULT_MODEL_NAME,
    BASE_DIR,
    TRT_BATCH_SIZE,
)
from src.core.loader import ThreadedStreamLoader
from src.core.image_pipeline import (
    collect_image_paths,
    prepare_image_for_buffer,
    restore_original_coordinates,
)
from src.core.runtime_reporting import write_runtime_report
from src.core.shared_memory_utils import cleanup_shared_memory, create_shared_buffer
from src.core.video_pipeline import (
    build_video_output_paths,
    is_stream_source,
    normalize_queue_controls,
    serialize_frame_detections,
    should_pause_dispatch,
)
from src.core.worker_io import dispatch_to_first_available_worker
from src.utils.visualization import draw_visuals
from src.utils.export import export_tensorrt

# Worker Runner Script Path
WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "worker_runner.py"
)


class VisualizerThread:
    def __init__(self, out_writer=None, shm_info=None, queue_size=128):
        self.q = queue.Queue(maxsize=queue_size)
        self.out_writer = out_writer
        self.stopped = False
        self.total_time = 0.0
        self.processed = 0
        self.shm_info = shm_info
        self.shm = None
        self.buffer_array = None

        if shm_info:
            self.shm = SharedMemory(name=shm_info["name"])
            self.buffer_array = np.ndarray(
                shm_info["shape"], dtype=np.uint8, buffer=self.shm.buf
            )

        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def run(self):
        while not self.stopped or not self.q.empty():
            try:
                item = self.q.get(timeout=0.1)
                frame_idx, final_boxes = item

                if self.out_writer and self.buffer_array is not None:
                    t0 = time.time()
                    # Modulo access to ring buffer
                    slot_idx = frame_idx % self.shm_info["shape"][0]
                    # We MUST copy here because the writer might be slow
                    # while the producer keeps rotating the ring buffer
                    frame = self.buffer_array[slot_idx].copy()

                    draw_visuals(frame, final_boxes)
                    self.out_writer.write(frame)
                    self.total_time += time.time() - t0
                    self.processed += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Visualizer] Error: {e}")
                continue

        if self.shm:
            self.shm.close()

    def stop(self):
        self.stopped = True
        self.thread.join()


class ASAP:
    def __init__(
        self,
        model_path=None,
        device="0,1",
        patch_size=PATCH_SIZE,
        min_overlap=MIN_OVERLAP,
        num_workers_per_gpu=2,
        batch_size=8,
        global_context=False,
        global_size=640,
    ):
        # Determine model path if not provided
        if model_path is None:
            # Try to find/build engine for current patch size
            engine_name = f"yolo11n_{patch_size}.engine"
            model_path = os.path.join(BASE_DIR, "models", engine_name)

            if not os.path.exists(model_path):
                print(
                    f"Engine not found at {model_path}. Attempting to build from {DEFAULT_MODEL_NAME}..."
                )
                os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
                # Export on the first available device from the device list
                first_device = device.split(",")[0] if isinstance(device, str) else "0"
                built_path = export_tensorrt(
                    DEFAULT_MODEL_NAME,
                    device=first_device,
                    imgsz=patch_size,
                    batch=max(batch_size, TRT_BATCH_SIZE),
                )
                if built_path:
                    # Rename to include patch size if export_tensorrt uses standard name
                    if built_path != model_path and os.path.exists(built_path):
                        os.rename(built_path, model_path)
                else:
                    raise FileNotFoundError(
                        f"Could not build engine at {model_path} and none provided."
                    )

        self.model_path = model_path
        self.device_list = self._parse_device_list(device)
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.num_workers_per_gpu = num_workers_per_gpu
        self.batch_size = batch_size
        self.global_context = global_context
        self.global_size = global_size

        self.workers = []
        self.worker_threads = []
        self.out_q = queue.Queue(maxsize=256)  # Increased queue size
        self._worker_reader_error_counts = {}
        self._worker_stderr_files = []

        if self.device_list == ["cpu"] and self.num_workers_per_gpu > 1:
            print("CUDA is unavailable; using a single CPU worker for smoke/demo runs.")
            self.num_workers_per_gpu = 1

        print(
            f"Initializing Real-Time Pipeline on devices: {self.device_list} "
            f"({self.num_workers_per_gpu} workers/device)"
        )

    def _worker_log_hint(self):
        log_dir = os.path.dirname(self.model_path)
        return os.path.join(log_dir, "worker_gpu*_slot*.stderr.log")

    def _parse_device_list(self, device):
        if not torch.cuda.is_available():
            return ["cpu"]
        available_count = torch.cuda.device_count()
        if device is None or str(device).lower() == "all":
            return list(range(available_count))
        try:
            requested_ids = [int(i.strip()) for i in str(device).split(",")]
        except:
            requested_ids = [0]
        return [i for i in requested_ids if i < available_count]

    def _reader_thread(self, proc, device_id):
        """Reads stdout from worker process and pushes to out_q"""
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            try:
                parts = line.strip().split("|", 4)
                if len(parts) == 5:
                    idx = int(parts[0])
                    total_latency = float(parts[1])
                    inf_latency = float(parts[2])
                    nms_latency = float(parts[3])
                    boxes = json.loads(parts[4])
                    self.out_q.put(
                        (idx, boxes, total_latency, inf_latency, nms_latency)
                    )
                elif len(parts) == 3:
                    idx = int(parts[0])
                    latency = float(parts[1])
                    boxes = json.loads(parts[2])
                    self.out_q.put((idx, boxes, latency, latency, 0.0))
                elif len(parts) == 2:  # Legacy fallback
                    idx = int(parts[0])
                    boxes = json.loads(parts[1])
                    self.out_q.put((idx, boxes, 0.0, 0.0, 0.0))
            except ValueError:
                err_count = self._worker_reader_error_counts.get(device_id, 0) + 1
                self._worker_reader_error_counts[device_id] = err_count
                if err_count <= 5:
                    print(
                        f"[Reader GPU {device_id}] Ignored malformed worker line: {line.strip()[:200]}"
                    )
            except Exception as e:
                err_count = self._worker_reader_error_counts.get(device_id, 0) + 1
                self._worker_reader_error_counts[device_id] = err_count
                if err_count <= 5:
                    print(f"[Reader GPU {device_id}] Error: {e}")

    def _start_workers(
        self,
        classes=None,
        conf=CONF_THRES,
        iou_thres=None,
        augment=False,
        shm_info=None,
        global_shm_info=None,
    ):
        self.out_q = queue.Queue(maxsize=128)
        self.workers = []
        self.worker_threads = []

        shm_shape_str = ",".join(map(str, shm_info["shape"]))

        # TensorRT/Ultralytics engine contexts can fail when multiple worker
        # processes deserialize the same plan file concurrently. Give every GPU
        # worker its own temporary copy; CPU smoke mode does not need copies.
        import shutil

        for dev_id in self.device_list:
            # Strict GPU isolation when CUDA is available. CPU smoke/demo runs
            # intentionally avoid CUDA_VISIBLE_DEVICES and execute on CPU.
            worker_env = os.environ.copy()
            if dev_id != "cpu":
                worker_env["CUDA_VISIBLE_DEVICES"] = str(dev_id)

            # Launch multiple workers per GPU
            for worker_id_in_gpu in range(self.num_workers_per_gpu):
                use_model = self._worker_model_path(dev_id, worker_id_in_gpu, shutil)
                cmd = [
                    sys.executable,
                    WORKER_SCRIPT,
                    "--device",
                    "cpu" if dev_id == "cpu" else "0",  # Isolated GPU view or CPU smoke mode
                    "--model",
                    use_model,
                    "--shm-name",
                    shm_info["name"],
                    "--shm-shape",
                    shm_shape_str,
                    "--patch-size",
                    str(self.patch_size),
                    "--min-overlap",
                    str(self.min_overlap),
                    "--batch-size",
                    str(self.batch_size),
                    "--conf",
                    str(conf),
                    "--worker-slot",
                    str(worker_id_in_gpu),
                ]
                if iou_thres is not None:
                    cmd.extend(["--iou-thres", str(iou_thres)])
                if classes is not None:
                    cmd.extend(["--classes", ",".join(str(cls) for cls in classes)])
                if augment:
                    cmd.append("--augment")
                if self.global_context:
                    cmd.append("--global-context")
                    if global_shm_info:
                        cmd.extend(["--global-shm-name", str(global_shm_info["name"])])
                        cmd.extend(["--global-size", str(self.global_size)])

                stderr_path = os.path.join(
                    os.path.dirname(self.model_path),
                    f"worker_gpu{dev_id}_slot{worker_id_in_gpu}.stderr.log",
                )
                stderr_file = open(stderr_path, "w", buffering=1)
                self._worker_stderr_files.append(stderr_file)

                p = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    env=worker_env,  # Apply Isolation
                    text=True,
                    bufsize=1,  # Line buffered
                )
                self.workers.append(p)

                # Start Reader Thread
                t = threading.Thread(
                    target=self._reader_thread, args=(p, dev_id), daemon=True
                )
                t.start()
                self.worker_threads.append(t)

                try:
                    self._wait_for_worker_ready(p, stderr_path)
                except Exception:
                    self._stop_workers()
                    raise


    def _raise_if_worker_exited(self):
        exited = [proc for proc in self.workers if proc.poll() is not None]
        if not exited:
            return
        codes = [proc.returncode for proc in exited]
        raise RuntimeError(
            "Worker process exited during inference; benchmark results are invalid. "
            f"Exit codes: {codes}. Check worker stderr logs under {self._worker_log_hint()}"
        )

    def _wait_for_worker_ready(self, proc, stderr_path, timeout_s=90.0):
        deadline = time.time() + timeout_s
        last_log = ""
        while time.time() < deadline:
            try:
                with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                    last_log = f.read()
            except OSError:
                last_log = ""

            if "Ready." in last_log:
                return
            if "Model Error:" in last_log:
                raise RuntimeError(
                    "Worker failed during model warmup. "
                    f"See {stderr_path}. Last log:\n{last_log[-2000:]}"
                )
            if proc.poll() is not None:
                raise RuntimeError(
                    "Worker exited before becoming ready. "
                    f"See {stderr_path}. Last log:\n{last_log[-2000:]}"
                )
            time.sleep(0.05)

        raise TimeoutError(
            "Timed out waiting for worker warmup. "
            f"See {stderr_path}. Last log:\n{last_log[-2000:]}"
        )

    def _worker_model_path(self, dev_id, worker_id_in_gpu, shutil_module):
        if dev_id == "cpu" or not str(self.model_path).endswith(".engine"):
            return self.model_path

        base, ext = os.path.splitext(self.model_path)
        copy_path = f"{base}_worker_gpu{dev_id}_slot{worker_id_in_gpu}{ext}"
        if not os.path.exists(copy_path):
            print(
                f"Creating engine copy for Worker {dev_id}:{worker_id_in_gpu}: {copy_path}"
            )
            shutil_module.copy2(self.model_path, copy_path)
        return copy_path

    def _stop_workers(self):
        for p in self.workers:
            if p.poll() is None:
                p.terminate()

        deadline = time.time() + 3.0
        alive = [p for p in self.workers if p.poll() is None]
        while alive and time.time() < deadline:
            time.sleep(0.05)
            alive = [p for p in self.workers if p.poll() is None]

        for p in alive:
            try:
                p.kill()
            except Exception:
                pass

        for p in self.workers:
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    if p.stdin:
                        p.stdin.close()
                except Exception:
                    pass
                try:
                    if p.stdout:
                        p.stdout.close()
                except Exception:
                    pass
        self.workers = []
        for f in self._worker_stderr_files:
            try:
                f.close()
            except Exception:
                pass
        self._worker_stderr_files = []

        # Cleanup temporary engine copies
        try:
            dir_path = os.path.dirname(self.model_path)
            for f in os.listdir(dir_path):
                if "_worker" in f and f.endswith(".engine"):
                    full_path = os.path.join(dir_path, f)
                    try:
                        os.remove(full_path)
                    except OSError:
                        pass
        except Exception:
            pass

    def predict_video(
        self,
        input_path,
        output_path,
        save=True,
        save_json=False,
        resize=None,
        classes=None,
        conf_thres=CONF_THRES,
        iou_thres=None,
        augment=False,
        crop_mode=False,
        plot_fps=False,
        bounded_latency=False,
        max_in_flight=64,
        drop_policy="drop_oldest",
        input_fps_cap=None,
        **kwargs,
    ):
        """
        Unified video/stream predictor.
        Automatically handles Real-time vs File.
        """
        is_stream = is_stream_source(input_path)
        num_frames = kwargs.get("num_frames")
        max_in_flight, drop_policy, input_fps_cap = normalize_queue_controls(
            max_in_flight,
            drop_policy,
            input_fps_cap,
        )
        filename, save_path, json_save_path = build_video_output_paths(
            input_path,
            output_path,
            is_stream=is_stream,
            save=save,
            save_json=save_json,
        )

        json_results = []

        if save or save_json:
            os.makedirs(output_path, exist_ok=True)

        RING_BUFFER_SIZE = 128

        # Determine Frame Shape for Buffer
        if resize:
            buf_h, buf_w = resize[1], resize[0]
        else:
            tmp_cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
            if not tmp_cap.isOpened():
                raise ValueError("Cannot open input")
            buf_w = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            buf_h = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tmp_cap.release()

        print(
            f"Allocating Shared Ring Buffer (Named): {RING_BUFFER_SIZE}x{buf_h}x{buf_w}x3"
        )
        main_buffer = create_shared_buffer((RING_BUFFER_SIZE, buf_h, buf_w, 3))
        self.shm = main_buffer.shm
        shared_ring_buffer_cpu = main_buffer.tensor

        # Potential Secondary Buffer for Global Resized frames
        self.global_shm = None
        global_shared_buffer_cpu = None
        global_shm_info = None

        if self.global_context:
            g_size = self.global_size
            print(
                f"Allocating Global Context Buffer: {RING_BUFFER_SIZE}x{g_size}x{g_size}x3"
            )
            global_buffer = create_shared_buffer((RING_BUFFER_SIZE, g_size, g_size, 3))
            self.global_shm = global_buffer.shm
            global_shared_buffer_cpu = global_buffer.tensor
            global_shm_info = {
                "name": self.global_shm.name,
                "shape": (RING_BUFFER_SIZE, g_size, g_size, 3),
            }

        loader = ThreadedStreamLoader(
            input_path,
            patch_size=self.patch_size,
            min_overlap=self.min_overlap,
            target_size=resize,
            global_size=self.global_size,
            queue_size=32,
            crop_mode=crop_mode,
            drop_if_full=(is_stream or input_fps_cap is not None),
            source_fps_cap=input_fps_cap,
            shared_buffer=shared_ring_buffer_cpu,  # Pass Tensor wrapper to Loader
            global_shared_buffer=global_shared_buffer_cpu,
        ).start()

        shm_info = main_buffer.info

        try:
            self._start_workers(
                classes, conf_thres, iou_thres, augment, shm_info, global_shm_info
            )
        except Exception:
            loader.stop()
            cleanup_shared_memory(getattr(self, "shm", None))
            cleanup_shared_memory(getattr(self, "global_shm", None))
            self.shm = None
            self.global_shm = None
            raise

        viz = None
        out = None
        if save:
            os.makedirs(output_path, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                save_path, fourcc, loader.fps_video, (loader.w, loader.h)
            )
            print(f"Saving output to {save_path}")
            viz = VisualizerThread(out, shm_info=shm_info, queue_size=128).start()

        pbar = tqdm(
            total=loader.total_frames if not is_stream else None,
            desc="Processing",
            unit="frame",
        )
        start_time = time.time()

        frames_buffered = {}  # index -> frame
        results_buffered = {}  # index -> boxes

        current_idx = 0
        next_emit_idx = 0
        frames_written = 0
        worker_ptr = 0

        # Profiling Data
        total_worker_times = []
        inf_times = []
        nms_times = []
        fps_history = []  # (time, fps)
        queue_depth_trace = []
        dispatch_times = {}
        dispatch_backlog = {}
        dispatch_to_result_ms = []
        dispatch_to_write_ms = []
        frame_latency_records = []
        dropped_inflight_indices = set()
        source_idx_by_dispatch = {}
        max_source_idx_seen = -1
        staleness_frames = []
        staleness_ms = []
        frames_seen = 0
        frames_dropped = 0
        dropped_by_policy = {"drop_oldest": 0, "drop_newest": 0, "latest_only": 0}
        reached_frame_limit = False
        last_fps_check = start_time
        last_queue_trace = start_time
        processed_since_check = 0

        try:
            while loader.more() or frames_buffered:
                # 1. Fill Worker Queues (Dispatch)
                while loader.more():
                    in_flight = current_idx - next_emit_idx
                    if should_pause_dispatch(
                        in_flight=in_flight,
                        max_in_flight=max_in_flight,
                    ):
                        break
                    try:
                        item = loader.q.get(timeout=0.01)
                    except queue.Empty:
                        break  # Go to result collection if no frames ready yet

                    if item is None:
                        break
                    frames_seen += 1
                    source_idx, frame_ref, _ = item
                    max_source_idx_seen = max(max_source_idx_seen, int(source_idx))

                    if (
                        bounded_latency
                        and (current_idx - next_emit_idx) >= max_in_flight
                    ):
                        if drop_policy == "drop_newest":
                            frames_dropped += 1
                            dropped_by_policy[drop_policy] += 1
                            continue
                        oldest_idx = (
                            min(frames_buffered.keys()) if frames_buffered else None
                        )
                        if oldest_idx is not None:
                            frames_buffered.pop(oldest_idx, None)
                            results_buffered.pop(oldest_idx, None)
                            dispatch_times.pop(oldest_idx, None)
                            dispatch_backlog.pop(oldest_idx, None)
                            source_idx_by_dispatch.pop(oldest_idx, None)
                            dropped_inflight_indices.add(int(oldest_idx))
                            frames_dropped += 1
                            dropped_by_policy[drop_policy] += 1
                        else:
                            frames_dropped += 1
                            dropped_by_policy[drop_policy] += 1
                            continue
                    if isinstance(frame_ref, (int, np.integer)):
                        # ZERO COPY: Just use the index from SHM
                        # We don't copy the 25MB frame in the main thread anymore
                        frame_to_dispatch = int(frame_ref)
                    elif isinstance(frame_ref, torch.Tensor):
                        # Still fallback for direct-tensor-pointers if used
                        frame_to_dispatch = frame_ref.cpu().numpy().copy()
                    else:
                        frame_to_dispatch = np.array(frame_ref, copy=True)

                    # Try to put into the next worker's stdin
                    pushed = False
                    while not pushed:
                        for _ in range(len(self.workers)):
                            target_worker_idx = worker_ptr % len(self.workers)
                            worker_ptr += 1

                            proc = self.workers[target_worker_idx]
                            if proc.poll() is not None:
                                continue

                            try:
                                proc.stdin.write(f"{current_idx}\n")
                                proc.stdin.flush()
                                # Only store index or reference, not full frame copy if possible
                                frames_buffered[current_idx] = frame_to_dispatch
                                dispatch_times[current_idx] = time.time()
                                dispatch_backlog[current_idx] = (
                                    current_idx - next_emit_idx
                                )
                                source_idx_by_dispatch[current_idx] = int(source_idx)
                                current_idx += 1
                                pushed = True
                                break
                            except Exception:
                                continue

                        if not pushed:
                            time.sleep(0.001)
                            if all(p.poll() is not None for p in self.workers):
                                raise RuntimeError(
                                    "All worker processes exited before accepting new work. "
                                    f"Check worker stderr logs under {self._worker_log_hint()}"
                                )

                # 2. Collect Results
                while not self.out_q.empty():
                    item = self.out_q.get_nowait()
                    recv_time = time.time()
                    if len(item) == 5:
                        idx, boxes, total_latency, inf_latency, nms_latency = item
                        total_worker_times.append(total_latency)
                        inf_times.append(inf_latency)
                        nms_times.append(nms_latency)
                    elif len(item) == 3:
                        idx, boxes, latency = item
                        total_worker_times.append(latency)
                        inf_times.append(latency)
                    else:
                        idx, boxes = item

                    d_ts = dispatch_times.get(idx)
                    d_backlog = dispatch_backlog.get(idx, 0)
                    if idx in dropped_inflight_indices:
                        continue
                    if d_ts is not None:
                        d2r_ms = (recv_time - d_ts) * 1000.0
                        dispatch_to_result_ms.append(d2r_ms)
                        frame_latency_records.append(
                            {
                                "frame_idx": int(idx),
                                "dispatch_backlog": int(d_backlog),
                                "dispatch_to_result_ms": float(d2r_ms),
                                "timestamp_s": float(recv_time - start_time),
                            }
                        )

                    results_buffered[idx] = boxes

                self._raise_if_worker_exited()

                # 3. Write/Display in order
                while next_emit_idx in dropped_inflight_indices:
                    dropped_inflight_indices.remove(next_emit_idx)
                    dispatch_times.pop(next_emit_idx, None)
                    dispatch_backlog.pop(next_emit_idx, None)
                    frames_buffered.pop(next_emit_idx, None)
                    results_buffered.pop(next_emit_idx, None)
                    source_idx_by_dispatch.pop(next_emit_idx, None)
                    next_emit_idx += 1

                while next_emit_idx in results_buffered:
                    boxes = results_buffered.pop(next_emit_idx)
                    frame_data = frames_buffered.pop(next_emit_idx)

                    if viz:
                        # Pass (index, boxes) to VisualizerThread
                        viz.q.put((next_emit_idx, boxes))

                    if save_json:
                        # boxes is likely a list of [x1, y1, x2, y2, conf, cls]
                        # We convert it to a serializable format (floats/ints)
                        json_results.append(
                            serialize_frame_detections(next_emit_idx, boxes)
                        )

                    pbar.update(1)
                    source_idx = source_idx_by_dispatch.pop(
                        next_emit_idx, next_emit_idx
                    )
                    stale_frames = max(0, int(max_source_idx_seen - source_idx))
                    staleness_frames.append(stale_frames)
                    fps_ref = (
                        loader.fps_video
                        if loader.fps_video and loader.fps_video > 0
                        else 30.0
                    )
                    staleness_ms.append((1000.0 * stale_frames) / fps_ref)
                    frames_written += 1
                    just_emitted_idx = next_emit_idx
                    next_emit_idx += 1

                    d_ts = dispatch_times.pop(just_emitted_idx, None)
                    d_backlog = dispatch_backlog.pop(just_emitted_idx, 0)
                    if d_ts is not None:
                        d2w_ms = (time.time() - d_ts) * 1000.0
                        dispatch_to_write_ms.append(d2w_ms)
                        frame_latency_records.append(
                            {
                                "frame_idx": int(just_emitted_idx),
                                "dispatch_backlog": int(d_backlog),
                                "dispatch_to_write_ms": float(d2w_ms),
                                "timestamp_s": float(time.time() - start_time),
                            }
                        )

                    if num_frames and frames_written >= num_frames:
                        print(f"\nReached frame limit: {num_frames}")
                        reached_frame_limit = True
                        break
                    processed_since_check += 1

                if reached_frame_limit:
                    break

                # FPS Check every 1s; queue-depth trace is diagnostic only, so
                # sample it instead of allocating a dict on every scheduler loop.
                now = time.time()
                if now - last_queue_trace >= 0.25:
                    queue_depth_trace.append(
                        {
                            "t_s": float(now - start_time),
                            "loader_q": int(loader.q.qsize()),
                            "loader_raw_q": int(loader.raw_q.qsize()),
                            "worker_out_q": int(self.out_q.qsize()),
                            "frames_buffered": int(len(frames_buffered)),
                            "results_buffered": int(len(results_buffered)),
                            "dispatch_backlog": int(current_idx - next_emit_idx),
                            "viz_q": int(viz.q.qsize()) if viz is not None else 0,
                        }
                    )
                    last_queue_trace = now

                if now - last_fps_check >= 1.0:
                    fps = processed_since_check / (now - last_fps_check)
                    fps_history.append((now - start_time, fps))
                    last_fps_check = now
                    processed_since_check = 0

                if not loader.more() and not frames_buffered:
                    break
                time.sleep(0.001)

            # Final collection of any remaining results in out_q. Avoid a fixed
            # post-run sleep when all dispatched frames have already been emitted;
            # otherwise benchmark FPS includes teardown latency rather than runtime.
            if frames_buffered or current_idx != next_emit_idx:
                time.sleep(0.05)
            while not self.out_q.empty():
                item = self.out_q.get_nowait()
                recv_time = time.time()
                if len(item) == 5:
                    idx, boxes, total_latency, inf_latency, nms_latency = item
                    total_worker_times.append(total_latency)
                    inf_times.append(inf_latency)
                    nms_times.append(nms_latency)
                elif len(item) == 3:
                    idx, boxes, latency = item
                    total_worker_times.append(latency)
                    inf_times.append(latency)
                else:
                    idx, boxes = item

                d_ts = dispatch_times.get(idx)
                d_backlog = dispatch_backlog.get(idx, 0)
                if idx in dropped_inflight_indices:
                    continue
                if d_ts is not None:
                    d2r_ms = (recv_time - d_ts) * 1000.0
                    dispatch_to_result_ms.append(d2r_ms)
                    frame_latency_records.append(
                        {
                            "frame_idx": int(idx),
                            "dispatch_backlog": int(d_backlog),
                            "dispatch_to_result_ms": float(d2r_ms),
                            "timestamp_s": float(recv_time - start_time),
                        }
                    )

                results_buffered[idx] = boxes

        except KeyboardInterrupt:
            print("\nStopped by User")
        finally:
            self._stop_workers()
            if viz:
                viz.stop()
            loader.stop()
            if out:
                out.release()

            # Save JSON results if requested
            if save_json and json_save_path:
                print(f"Saving JSON results to {json_save_path}")
                with open(json_save_path, "w") as f:
                    json.dump(json_results, f, indent=2)

            pbar.close()

            # Cleanup Shared Memory
            cleanup_shared_memory(getattr(self, "shm", None))
            cleanup_shared_memory(getattr(self, "global_shm", None))
            self.shm = None
            self.global_shm = None

            total_time = time.time() - start_time
            write_runtime_report(
                output_path=output_path,
                plot_fps=plot_fps,
                frames_written=frames_written,
                total_time=total_time,
                fps_history=fps_history,
                total_worker_times=total_worker_times,
                inf_times=inf_times,
                nms_times=nms_times,
                loader_time_read=loader.time_read,
                loader_time_preprocess=loader.time_preprocess,
                loader_frame_count=loader.frame_count,
                loader_fps_video=loader.fps_video,
                visual_total_time=viz.total_time if viz else 0.0,
                visual_processed=viz.processed if viz else 0,
                frames_seen=frames_seen,
                frames_dropped=frames_dropped,
                input_fps_cap=input_fps_cap,
                bounded_latency=bounded_latency,
                max_in_flight=max_in_flight,
                drop_policy=drop_policy,
                dropped_by_policy=dropped_by_policy,
                dispatch_to_result_ms=dispatch_to_result_ms,
                dispatch_to_write_ms=dispatch_to_write_ms,
                queue_depth_trace=queue_depth_trace,
                staleness_frames=staleness_frames,
                staleness_ms=staleness_ms,
                frame_latency_records=frame_latency_records,
            )

    def predict(
        self,
        image: np.ndarray,
        classes=None,
        conf_thres=CONF_THRES,
        iou_thres=None,
        augment=False,
    ):
        """
        SAHI-like single image prediction API.
        Takes a numpy array image and returns bounding boxes without disk I/O.
        """
        # 1. Ensure workers and SHM are initialized
        if getattr(self, "shm", None) is None:
            # Default buffer size 4K
            BUF_H, BUF_W = 2160, 3840
            main_buffer = create_shared_buffer((1, BUF_H, BUF_W, 3))
            self.shm = main_buffer.shm
            self._shm_array = main_buffer.array
            shm_info = main_buffer.info

            self.global_shm = None
            global_shm_info = None
            if self.global_context:
                g_size = self.global_size
                global_buffer = create_shared_buffer((1, g_size, g_size, 3))
                self.global_shm = global_buffer.shm
                self._global_shm_array = global_buffer.array
                global_shm_info = global_buffer.info

            self._start_workers(
                classes, conf_thres, iou_thres, augment, shm_info, global_shm_info
            )

        BUF_H, BUF_W = self._shm_array.shape[1:3]
        img_h, img_w = image.shape[:2]

        if getattr(self, "global_context", False):
            # Pre-resize for global view directly
            g_size = self.global_size
            g_img = cv2.resize(image, (g_size, g_size))
            self._global_shm_array[0] = g_img

        img_padded, ratio, pad, _img_h, _img_w = prepare_image_for_buffer(
            image,
            buffer_height=BUF_H,
            buffer_width=BUF_W,
        )

        # Write to Shared Memory
        self._shm_array[0] = img_padded

        # 3. Dispatch to worker
        dispatch_to_first_available_worker(self.workers, 0)

        # 4. Wait for Result
        try:
            item = self.out_q.get(timeout=60.0)
            idx, boxes = item[0], item[1]
        except queue.Empty:
            print(f"Timeout processing image")
            return []

        # 5. Unpad & Rescale Coordinates
        return restore_original_coordinates(
            boxes,
            ratio=ratio,
            pad=pad,
            image_width=img_w,
            image_height=img_h,
        )

    def cleanup(self):
        """Clean up shared memory and worker processes."""
        self._stop_workers()
        cleanup_shared_memory(getattr(self, "shm", None))
        cleanup_shared_memory(getattr(self, "global_shm", None))
        self.shm = None
        self.global_shm = None

    def __del__(self):
        self.cleanup()

    def predict_image(
        self,
        input_path,
        output_path,
        save=True,
        resize=None,
        classes=None,
        conf_thres=CONF_THRES,
        iou_thres=None,
        augment=False,
    ):
        """
        Process a single image or directory of images.
        """
        # 1. Gather Images
        images = collect_image_paths(input_path)

        print(f"Processing {len(images)} images...")
        if save:
            os.makedirs(output_path, exist_ok=True)

        # 2. Setup Worker Pipeline (Assume max 4K for buffer)
        # We process one by one, so buffer size 1 is enough
        BUF_H, BUF_W = 2160, 3840
        if resize:
            BUF_H, BUF_W = resize[1], resize[0]

        main_buffer = create_shared_buffer((1, BUF_H, BUF_W, 3))
        self.shm = main_buffer.shm
        shm_array = main_buffer.array
        shm_info = main_buffer.info

        self._start_workers(classes, conf_thres, iou_thres, augment, shm_info)

        results = {}

        try:
            for img_path in tqdm(images, desc="Inference"):
                # Read
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img, ratio, pad, img_h, img_w = prepare_image_for_buffer(
                    img,
                    buffer_height=BUF_H,
                    buffer_width=BUF_W,
                    resize=resize,
                )

                # Write to Shared Memory (Index 0)
                shm_array[0] = img

                # Dispatch
                dispatch_to_first_available_worker(self.workers, 0)

                # Wait for Result
                try:
                    # Increase timeout significantly for the first image (warmup)
                    # or just use a larger default for predict_image
                    item = self.out_q.get(timeout=60.0)
                    idx, boxes = item[0], item[1]
                except queue.Empty:
                    print(f"Timeout processing {img_path}")
                    continue

                # Save/Visualize
                # Rescale boxes back to original image coordinates
                # boxes format: [x1, y1, x2, y2, conf, cls] relative to PADDED image
                # x_original = (x_padded - dw) / ratio_w
                # y_original = (y_padded - dh) / ratio_h

                boxes = restore_original_coordinates(
                    boxes,
                    ratio=ratio,
                    pad=pad,
                    image_width=img_w,
                    image_height=img_h,
                )

                results[img_path] = boxes

                if save:
                    draw_visuals(img, boxes)
                    save_name = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(output_path, save_name), img)

        finally:
            self._stop_workers()
            cleanup_shared_memory(getattr(self, "shm", None))
            cleanup_shared_memory(getattr(self, "global_shm", None))
            self.shm = None
            self.global_shm = None

        return results
