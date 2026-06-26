import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.core.runtime_reporting import write_runtime_report


class RuntimeReportingTest(unittest.TestCase):
    def _report_kwargs(self, output_path, *, frames_written=11):
        return dict(
            output_path=output_path,
            plot_fps=False,
            frames_written=frames_written,
            total_time=1.1,
            fps_history=[(float(i), 10.0) for i in range(11)],
            total_worker_times=[0.02, 0.03],
            inf_times=[0.01, 0.02],
            nms_times=[0.001, 0.002],
            loader_time_read=0.11,
            loader_time_preprocess=0.22,
            loader_frame_count=11,
            loader_fps_video=30.0,
            visual_total_time=0.055,
            visual_processed=11,
            frames_seen=12,
            frames_dropped=1,
            input_fps_cap=None,
            bounded_latency=True,
            max_in_flight=4,
            drop_policy="drop_oldest",
            dropped_by_policy={"drop_oldest": 1, "drop_newest": 0, "latest_only": 0},
            dispatch_to_result_ms=[10.0, 20.0],
            dispatch_to_write_ms=[12.0, 24.0],
            queue_depth_trace=[{"dispatch_backlog": 1}, {"dispatch_backlog": 2}],
            staleness_frames=[0, 1],
            staleness_ms=[0.0, 33.3],
            frame_latency_records=[
                {"dispatch_backlog": 1, "dispatch_to_write_ms": 12.0}
            ],
        )

    def test_report_writes_latency_trace_without_plotting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = write_runtime_report(**self._report_kwargs(tmp_dir))

            trace_path = Path(tmp_dir) / "latency_trace_detailed.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertAlmostEqual(result["avg_fps"], 10.0)
        self.assertIn("LATENCY PROFILE SUMMARY", stdout.getvalue())
        self.assertEqual(trace["summary"]["queue_control"]["frames_dropped"], 1)
        self.assertEqual(trace["summary"]["dispatch_to_write_ms"]["count"], 2)

    def test_short_runs_skip_detailed_trace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = write_runtime_report(
                    **self._report_kwargs(tmp_dir, frames_written=10)
                )

            self.assertFalse((Path(tmp_dir) / "latency_trace_detailed.json").exists())
        self.assertAlmostEqual(result["avg_fps"], 10 / 1.1)


if __name__ == "__main__":
    unittest.main()
