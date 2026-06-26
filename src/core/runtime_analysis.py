from __future__ import annotations

import numpy as np


def percentile_summary(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_stable_fps_range(
    fps_history,
    *,
    min_stable_seconds=5,
    min_history_seconds=20.0,
    min_post_warmup_samples=5,
    warmup_cv_tolerance=0.15,
    warmup_band_tolerance=0.15,
    min_significant_fps_ratio=0.30,
    tail_spike_tolerance=0.35,
    tail_baseline_samples=5,
):
    result = {
        "stable_start_idx": 0,
        "stable_end_idx": len(fps_history),
        "stable_fps_list": [],
    }
    if not fps_history:
        return result

    fps_values = np.array([fps for _time_s, fps in fps_history], dtype=float)
    time_values = np.array([time_s for time_s, _fps in fps_history], dtype=float)
    history_duration = (
        float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0
    )

    # Short smoke runs do not contain enough evidence to infer a human-like
    # warmup boundary. Keep a conservative non-empty slice instead of pretending
    # a stable region was discovered.
    if (
        len(fps_values) < min_post_warmup_samples
        or history_duration < min_history_seconds
    ):
        stable_start_idx = (
            0
            if len(fps_values) < min_post_warmup_samples
            else min(2, len(fps_values) - 1)
        )
        stable_end_idx = len(fps_values)
    else:
        stable_end_idx = len(fps_values)

        # Trim only contiguous tail drain/spike samples. Use the immediately
        # preceding local baseline so interior throughput changes are preserved.
        while stable_end_idx > min_post_warmup_samples + tail_baseline_samples:
            baseline_window = fps_values[
                stable_end_idx - tail_baseline_samples - 1 : stable_end_idx - 1
            ]
            baseline_window = baseline_window[baseline_window > 0]
            if baseline_window.size == 0:
                break
            baseline = float(np.median(baseline_window))
            last = float(fps_values[stable_end_idx - 1])
            if baseline <= 0:
                break
            lower = baseline * (1.0 - tail_spike_tolerance)
            upper = baseline * (1.0 + tail_spike_tolerance)
            if last < lower or last > upper:
                stable_end_idx -= 1
                continue
            break

        candidate_values = fps_values[:stable_end_idx]
        positive = candidate_values[candidate_values > 0]
        # Avoid selecting the initial all-zero or tiny ramp as stable. The
        # threshold is deliberately relative to the run's observed capability,
        # not a hard-coded FPS number.
        significant_floor = 0.0
        if positive.size:
            significant_floor = max(
                1.0,
                float(np.percentile(positive, 90)) * min_significant_fps_ratio,
            )

        stable_start_idx = 0
        for index in range(0, len(candidate_values)):
            end_index = min(len(candidate_values), index + min_stable_seconds)
            window = candidate_values[index:end_index]
            window_times = time_values[index:end_index]
            if window.size < min_post_warmup_samples:
                continue
            if window_times[-1] - window_times[0] < max(
                1.0, min_stable_seconds - 1
            ):
                continue
            window_median = float(np.median(window))
            if window_median <= significant_floor:
                continue
            cv = (
                float(np.std(window) / window_median)
                if window_median
                else float("inf")
            )
            lower = window_median * (1.0 - warmup_band_tolerance)
            upper = window_median * (1.0 + warmup_band_tolerance)
            in_band = (window >= lower) & (window <= upper)
            mostly_in_band = float(np.mean(in_band))
            # The first sample of the accepted window must already be in-band;
            # otherwise a final ramp sample can be incorrectly included as the
            # beginning of the stable region.
            if (
                cv <= warmup_cv_tolerance
                and mostly_in_band >= 0.8
                and bool(in_band[0])
            ):
                stable_start_idx = index
                break
        else:
            # Fallback: first meaningful sample after ramp.
            candidates = np.flatnonzero(candidate_values >= significant_floor)
            stable_start_idx = int(candidates[0]) if candidates.size else 0

    stable_end_idx = max(stable_start_idx + 1, stable_end_idx)
    stable_fps_list = fps_values[stable_start_idx:stable_end_idx].tolist()
    result.update(
        {
            "stable_start_idx": stable_start_idx,
            "stable_end_idx": stable_end_idx,
            "stable_fps_list": stable_fps_list,
        }
    )
    return result


def summarize_queue_control(
    *,
    frames_seen: int,
    frames_written: int,
    frames_dropped: int,
    input_fps_cap,
    loader_fps_video,
    bounded_latency: bool,
    max_in_flight: int,
    drop_policy: str,
    dropped_by_policy: dict,
):
    attempted = frames_seen if frames_seen > 0 else (frames_written + frames_dropped)
    retain_ratio = (frames_written / attempted) if attempted > 0 else 0.0
    if input_fps_cap is not None and input_fps_cap > 0:
        input_fps_ref = float(input_fps_cap)
    else:
        input_fps_ref = loader_fps_video if loader_fps_video and loader_fps_video > 0 else 30.0
    retained_fps = input_fps_ref * retain_ratio
    drop_rate = (100.0 * frames_dropped / attempted) if attempted > 0 else 0.0
    return {
        "attempted": int(attempted),
        "retain_ratio": float(retain_ratio),
        "input_fps_reference": float(input_fps_ref),
        "retained_fps": float(retained_fps),
        "drop_rate": float(drop_rate),
        "summary": {
            "bounded_latency": bool(bounded_latency),
            "max_in_flight": int(max_in_flight),
            "drop_policy": drop_policy,
            "input_fps_cap": float(input_fps_cap) if input_fps_cap else None,
            "input_fps_reference": float(input_fps_ref),
            "frames_seen": int(attempted),
            "frames_written": int(frames_written),
            "frames_dropped": int(frames_dropped),
            "drop_rate_percent": float(drop_rate),
            "retain_ratio": float(retain_ratio),
            "retained_fps_estimate": float(retained_fps),
            "dropped_by_policy": {
                "drop_oldest": int(dropped_by_policy["drop_oldest"]),
                "drop_newest": int(dropped_by_policy["drop_newest"]),
                "latest_only": int(dropped_by_policy["latest_only"]),
            },
        },
    }


def summarize_latency_trace(
    *,
    dispatch_to_result_ms,
    dispatch_to_write_ms,
    queue_depth_trace,
    staleness_frames,
    staleness_ms,
    frame_latency_records,
):
    d2r_stats = percentile_summary(dispatch_to_result_ms)
    d2w_stats = percentile_summary(dispatch_to_write_ms)
    backlog_values = [row["dispatch_backlog"] for row in queue_depth_trace]
    backlog_stats = percentile_summary(backlog_values)
    stale_frame_stats = percentile_summary(staleness_frames)
    stale_ms_stats = percentile_summary(staleness_ms)

    low_backlog = [
        rec["dispatch_to_write_ms"]
        for rec in frame_latency_records
        if "dispatch_to_write_ms" in rec and rec["dispatch_backlog"] <= 15
    ]
    mid_backlog = [
        rec["dispatch_to_write_ms"]
        for rec in frame_latency_records
        if "dispatch_to_write_ms" in rec and 16 <= rec["dispatch_backlog"] <= 31
    ]
    high_backlog = [
        rec["dispatch_to_write_ms"]
        for rec in frame_latency_records
        if "dispatch_to_write_ms" in rec and rec["dispatch_backlog"] >= 32
    ]

    return {
        "dispatch_to_result_ms": d2r_stats,
        "dispatch_to_write_ms": d2w_stats,
        "queue_backlog_frames": backlog_stats,
        "freshness_lag_frames": stale_frame_stats,
        "freshness_lag_ms": stale_ms_stats,
        "queue_conditioned_dispatch_to_write_ms": {
            "q_0_15": percentile_summary(low_backlog),
            "q_16_31": percentile_summary(mid_backlog),
            "q_32_plus": percentile_summary(high_backlog),
        },
    }
