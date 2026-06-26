from __future__ import annotations

import json
import os

import numpy as np

from src.core.runtime_analysis import (
    compute_stable_fps_range,
    summarize_latency_trace,
    summarize_queue_control,
)


def write_runtime_report(
    *,
    output_path: str | None,
    plot_fps: bool,
    frames_written: int,
    total_time: float,
    fps_history,
    total_worker_times,
    inf_times,
    nms_times,
    loader_time_read: float,
    loader_time_preprocess: float,
    loader_frame_count: int,
    loader_fps_video: float,
    visual_total_time: float,
    visual_processed: int,
    frames_seen: int,
    frames_dropped: int,
    input_fps_cap,
    bounded_latency: bool,
    max_in_flight: int,
    drop_policy: str,
    dropped_by_policy: dict,
    dispatch_to_result_ms,
    dispatch_to_write_ms,
    queue_depth_trace,
    staleness_frames,
    staleness_ms,
    frame_latency_records,
):
    avg_fps, stable_fps_data = _print_realtime_profile(
        frames_written=frames_written,
        total_time=total_time,
        fps_history=fps_history,
    )

    if frames_written <= 10:
        return {"avg_fps": float(avg_fps), "stable_fps": stable_fps_data}

    print("Generating Performance Report...")
    timings = _average_timings(
        total_worker_times=total_worker_times,
        inf_times=inf_times,
        nms_times=nms_times,
        loader_time_read=loader_time_read,
        loader_time_preprocess=loader_time_preprocess,
        loader_frame_count=loader_frame_count,
        visual_total_time=visual_total_time,
        visual_processed=visual_processed,
        avg_fps=avg_fps,
    )

    if plot_fps:
        _save_benchmark_plot(
            output_path=output_path,
            avg_fps=avg_fps,
            fps_history=fps_history,
            stable_fps_data=stable_fps_data,
            timings=timings,
        )

    _print_latency_summary(timings, avg_fps)
    queue_control = _print_queue_control(
        frames_seen=frames_seen,
        frames_written=frames_written,
        frames_dropped=frames_dropped,
        input_fps_cap=input_fps_cap,
        loader_fps_video=loader_fps_video,
        bounded_latency=bounded_latency,
        max_in_flight=max_in_flight,
        drop_policy=drop_policy,
        dropped_by_policy=dropped_by_policy,
    )
    latency_stats = _print_latency_percentiles(
        dispatch_to_result_ms=dispatch_to_result_ms,
        dispatch_to_write_ms=dispatch_to_write_ms,
        queue_depth_trace=queue_depth_trace,
        staleness_frames=staleness_frames,
        staleness_ms=staleness_ms,
        frame_latency_records=frame_latency_records,
    )
    latency_trace = _build_latency_trace(
        queue_control=queue_control,
        latency_stats=latency_stats,
        queue_depth_trace=queue_depth_trace,
        frame_latency_records=frame_latency_records,
        total_time=total_time,
        avg_fps=avg_fps,
    )
    _write_latency_trace(output_path, latency_trace)
    _print_bottleneck_analysis(timings, avg_fps)

    return {
        "avg_fps": float(avg_fps),
        "stable_fps": stable_fps_data,
        "timings": timings,
        "queue_control": queue_control,
        "latency_stats": latency_stats,
    }


def _print_realtime_profile(*, frames_written: int, total_time: float, fps_history):
    avg_fps = frames_written / total_time if total_time > 0 else 0.0
    print(
        f"\n[Real-Time Profile] Total Time: {total_time:.2f}s, "
        f"Processed: {frames_written} frames, Avg FPS: {avg_fps:.2f}"
    )

    stable_fps_data = compute_stable_fps_range(fps_history)
    stable_fps_list = stable_fps_data["stable_fps_list"]
    if stable_fps_list:
        stable_min, stable_max = min(stable_fps_list), max(stable_fps_list)
        stable_mean = sum(stable_fps_list) / len(stable_fps_list)
        print(
            f"[Stable FPS] Min: {stable_min:.2f}, Mean: {stable_mean:.2f}, "
            f"Max: {stable_max:.2f} (Calculated via Dynamic Range Check)"
        )
    return float(avg_fps), stable_fps_data


def _average_timings(
    *,
    total_worker_times,
    inf_times,
    nms_times,
    loader_time_read: float,
    loader_time_preprocess: float,
    loader_frame_count: int,
    visual_total_time: float,
    visual_processed: int,
    avg_fps: float,
):
    avg_read = loader_time_read / loader_frame_count if loader_frame_count > 0 else 0.0
    avg_preprocess = (
        loader_time_preprocess / loader_frame_count if loader_frame_count > 0 else 0.0
    )
    avg_visual = visual_total_time / visual_processed if visual_processed > 0 else 0.0
    return {
        "worker_total": float(np.mean(total_worker_times)) if total_worker_times else 0.0,
        "inference": float(np.mean(inf_times)) if inf_times else 0.0,
        "nms": float(np.mean(nms_times)) if nms_times else 0.0,
        "read": float(avg_read),
        "preprocess": float(avg_preprocess),
        "visual": float(avg_visual),
        "e2e_ms": float((1000.0 / avg_fps) if avg_fps > 0 else 0.0),
    }


def _save_benchmark_plot(*, output_path, avg_fps, fps_history, stable_fps_data, timings):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    _plot_fps_history(plt, avg_fps, fps_history, stable_fps_data)
    _plot_latency_breakdown(plt, timings)
    plt.tight_layout()

    plot_path = os.path.join(output_path, "benchmark_report.png")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Report saved to {plot_path}")


def _plot_fps_history(plt, avg_fps, fps_history, stable_fps_data):
    if not fps_history:
        return

    times, fps_vals = zip(*fps_history)
    times = np.array(times)
    fps_vals = np.array(fps_vals)
    stable_start_idx = stable_fps_data["stable_start_idx"]
    stable_end_idx = stable_fps_data["stable_end_idx"]
    stable_fps_list = stable_fps_data["stable_fps_list"]

    plt.subplot(2, 1, 1)
    plt.plot(
        times,
        fps_vals,
        color="lightgray",
        linestyle="-",
        linewidth=1,
        label="Warmup/Cooldown",
    )

    stable_mean = None
    if len(fps_history) >= 10 and stable_fps_list:
        stable_times = times[stable_start_idx:stable_end_idx]
        stable_vals = fps_vals[stable_start_idx:stable_end_idx]
        if len(stable_vals) > 0:
            stable_mean = _plot_stable_region(plt, stable_times, stable_vals)

    if stable_mean is not None:
        plt.title(f"Real-Time FPS Stability (Stable Avg: {stable_mean:.1f})")
    else:
        plt.title(f"Real-Time FPS Stability (Avg: {avg_fps:.1f})")
    plt.ylabel("FPS")
    plt.xlabel("Time (s)")
    plt.legend(loc="lower right", fontsize="small", ncol=2)
    plt.grid(True, alpha=0.5)


def _plot_stable_region(plt, stable_times, stable_vals):
    stable_mean = float(np.mean(stable_vals))
    stable_min = float(np.min(stable_vals))
    stable_max = float(np.max(stable_vals))

    plt.plot(stable_times, stable_vals, color="#1f77b4", linewidth=2, label="Stable Region")
    plt.axvspan(
        stable_times[0],
        stable_times[-1],
        color="#e6f2ff",
        alpha=0.3,
        label="Stable Zone",
    )
    plt.axhline(y=stable_mean, color="green", linestyle="--", linewidth=1.5, label=f"Mean: {stable_mean:.1f}")
    plt.axhline(y=stable_max, color="red", linestyle=":", alpha=0.7, label=f"Max: {stable_max:.1f}")
    plt.axhline(y=stable_min, color="orange", linestyle=":", alpha=0.7, label=f"Min: {stable_min:.1f}")

    max_t = stable_times[np.argmax(stable_vals)]
    min_t = stable_times[np.argmin(stable_vals)]
    plt.text(max_t, stable_max, f"{stable_max:.1f}", color="red", va="bottom", ha="center", fontsize=8, fontweight="bold")
    plt.text(min_t, stable_min, f"{stable_min:.1f}", color="orange", va="top", ha="center", fontsize=8, fontweight="bold")
    return stable_mean


def _plot_latency_breakdown(plt, timings):
    plt.subplot(2, 1, 2)
    labels = ["Worker Total", "Preprocess (CPU)", "Reader (CPU)"]
    values_ms = [
        timings["worker_total"] * 1000,
        timings["preprocess"] * 1000,
        timings["read"] * 1000,
    ]
    plt.barh(labels, values_ms, color=["blue", "orange", "green"])
    plt.title("Component Latency (ms)")
    plt.xlabel("Time (ms)")
    max_val = max(values_ms) if values_ms else 1
    plt.xlim(0, max_val * 1.15)
    for index, value in enumerate(values_ms):
        plt.text(value + (max_val * 0.01), index, f" {value:.1f}ms", va="center", ha="left")


def _print_latency_summary(timings, avg_fps):
    print("\n" + "=" * 80)
    print("LATENCY PROFILE SUMMARY (DETAILED)")
    print("=" * 80)
    print(f"Video I/O (Read):             {timings['read'] * 1000:>12.2f} ms")
    print(f"CPU Preprocess:               {timings['preprocess'] * 1000:>12.2f} ms")
    print(f"GPU Inference:                {timings['inference'] * 1000:>12.2f} ms")
    print(f"NMS Post-processing:          {timings['nms'] * 1000:>12.2f} ms")
    print(f"Visualization & Save:         {timings['visual'] * 1000:>12.2f} ms")
    print("-" * 80)
    print(f"Total (End-to-End):           {timings['e2e_ms']:>12.2f} ms")
    print(f"Average FPS:                  {avg_fps:>12.2f}")
    print(f"Estimated FPS (Real):        {avg_fps:>12.2f}")
    print("=" * 80)


def _print_queue_control(**kwargs):
    queue_control = summarize_queue_control(**kwargs)
    print("\n[Queue Control]")
    print(f" - Bounded Latency Mode: {bool(kwargs['bounded_latency'])}")
    print(f" - Max In-Flight Frames: {int(kwargs['max_in_flight'])}")
    print(f" - Drop Policy:          {kwargs['drop_policy']}")
    print(f" - Frames Seen:          {int(queue_control['attempted'])}")
    print(f" - Frames Written:       {int(kwargs['frames_written'])}")
    print(
        f" - Frames Dropped:       {int(kwargs['frames_dropped'])} "
        f"({queue_control['drop_rate']:.2f}%)"
    )
    print(f" - Retained FPS (est):   {queue_control['retained_fps']:.2f}")
    return queue_control


def _print_latency_percentiles(**kwargs):
    latency_stats = summarize_latency_trace(**kwargs)
    d2r_stats = latency_stats["dispatch_to_result_ms"]
    d2w_stats = latency_stats["dispatch_to_write_ms"]
    backlog_stats = latency_stats["queue_backlog_frames"]
    stale_frame_stats = latency_stats["freshness_lag_frames"]
    stale_ms_stats = latency_stats["freshness_lag_ms"]

    print("\n[End-to-End Percentiles]")
    print(_format_percentile_line("Dispatch->Result (ms)", d2r_stats))
    print(_format_percentile_line("Dispatch->Write  (ms)", d2w_stats))
    print(_format_percentile_line("Queue Backlog    (fr)", backlog_stats))
    print(_format_percentile_line("Freshness Lag    (fr)", stale_frame_stats))
    print(_format_percentile_line("Freshness Lag    (ms)", stale_ms_stats))
    return latency_stats


def _format_percentile_line(label, stats):
    return (
        f" - {label}: "
        f"p50={stats['p50']:.2f}, p95={stats['p95']:.2f}, p99={stats['p99']:.2f}"
    )


def _build_latency_trace(
    *,
    queue_control,
    latency_stats,
    queue_depth_trace,
    frame_latency_records,
    total_time,
    avg_fps,
):
    return {
        "summary": {
            "queue_control": {
                **queue_control["summary"],
                "run_wall_time_s": float(total_time),
                "observed_throughput_fps": float(avg_fps),
            },
            "dispatch_to_result_ms": latency_stats["dispatch_to_result_ms"],
            "dispatch_to_write_ms": latency_stats["dispatch_to_write_ms"],
            "queue_backlog_frames": latency_stats["queue_backlog_frames"],
            "freshness_lag_frames": latency_stats["freshness_lag_frames"],
            "freshness_lag_ms": latency_stats["freshness_lag_ms"],
            "queue_conditioned_dispatch_to_write_ms": latency_stats[
                "queue_conditioned_dispatch_to_write_ms"
            ],
        },
        "queue_depth_trace": queue_depth_trace,
        "frame_latency_records": frame_latency_records,
    }


def _write_latency_trace(output_path, latency_trace):
    if not output_path:
        return
    os.makedirs(output_path, exist_ok=True)
    latency_trace_path = os.path.join(output_path, "latency_trace_detailed.json")
    with open(latency_trace_path, "w", encoding="utf-8") as handle:
        json.dump(latency_trace, handle, indent=2)
    print(f"Latency trace saved to {latency_trace_path}")


def _print_bottleneck_analysis(timings, avg_fps):
    print("\n[Bottleneck Analysis]")
    print(f" - Latency (Per Worker): {timings['inference'] * 1000:.2f} ms")
    print(f" - System Throughput:    {avg_fps:.2f} FPS")
    print(f" - Avg Preprocess Time:  {timings['preprocess'] * 1000:.2f} ms")
    if timings["inference"] > timings["preprocess"] and timings["inference"] > timings["read"]:
        print(" -> Bottleneck: GPU Inference (Workers are busy)")
    else:
        print(" -> Bottleneck: CPU Preprocessing/Reading")
