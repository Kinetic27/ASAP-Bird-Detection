import unittest
import importlib.util

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("optional dependency numpy is not installed")


from src.core.runtime_analysis import (
    compute_stable_fps_range,
    percentile_summary,
    summarize_latency_trace,
    summarize_queue_control,
)


class RuntimeAnalysisTest(unittest.TestCase):
    def test_percentile_summary_handles_empty_values(self):
        summary = percentile_summary([])
        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["p95"], 0.0)

    def test_compute_stable_fps_range_detects_adaptive_warmup_and_trims_only_tail_spike(self):
        fps_values = [
            0,
            0,
            0,
            0,
            0,
            1,
            4,
            11,
            32,
            40,
            40,
            40,
            40,
            40,
            40,
            48,
            40,
            40,
            48,
            40,
            56,
            56,
            48,
            56,
            80,
        ]
        fps_history = [(float(i), fps) for i, fps in enumerate(fps_values)]

        result = compute_stable_fps_range(fps_history)

        self.assertGreaterEqual(result["stable_start_idx"], 9)
        self.assertEqual(result["stable_end_idx"], len(fps_values) - 1)
        self.assertNotIn(0, result["stable_fps_list"])
        self.assertNotIn(80, result["stable_fps_list"])
        self.assertIn(56, result["stable_fps_list"])

    def test_summarize_queue_control_uses_loader_fps_when_cap_missing(self):
        summary = summarize_queue_control(
            frames_seen=10,
            frames_written=8,
            frames_dropped=2,
            input_fps_cap=None,
            loader_fps_video=25.0,
            bounded_latency=True,
            max_in_flight=4,
            drop_policy="drop_oldest",
            dropped_by_policy={"drop_oldest": 2, "drop_newest": 0, "latest_only": 0},
        )

        self.assertEqual(summary["attempted"], 10)
        self.assertAlmostEqual(summary["retained_fps"], 20.0)
        self.assertEqual(summary["summary"]["drop_policy"], "drop_oldest")

    def test_summarize_latency_trace_builds_backlog_buckets(self):
        trace = summarize_latency_trace(
            dispatch_to_result_ms=[10, 20, 30],
            dispatch_to_write_ms=[15, 25, 35],
            queue_depth_trace=[{"dispatch_backlog": 1}, {"dispatch_backlog": 20}, {"dispatch_backlog": 40}],
            staleness_frames=[0, 1, 2],
            staleness_ms=[0.0, 33.0, 66.0],
            frame_latency_records=[
                {"dispatch_backlog": 5, "dispatch_to_write_ms": 10},
                {"dispatch_backlog": 20, "dispatch_to_write_ms": 20},
                {"dispatch_backlog": 40, "dispatch_to_write_ms": 30},
            ],
        )

        self.assertEqual(trace["dispatch_to_result_ms"]["count"], 3)
        self.assertEqual(trace["queue_conditioned_dispatch_to_write_ms"]["q_0_15"]["count"], 1)
        self.assertEqual(trace["queue_conditioned_dispatch_to_write_ms"]["q_16_31"]["count"], 1)
        self.assertEqual(trace["queue_conditioned_dispatch_to_write_ms"]["q_32_plus"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
