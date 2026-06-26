import unittest

from src.core.video_pipeline import (
    build_video_output_paths,
    is_stream_source,
    normalize_queue_controls,
    serialize_frame_detections,
    should_pause_dispatch,
)


class VideoPipelineTest(unittest.TestCase):
    def test_is_stream_source_recognizes_urls_and_camera_indices(self):
        self.assertTrue(is_stream_source("rtsp://cam"))
        self.assertTrue(is_stream_source("0"))
        self.assertFalse(is_stream_source("video.mp4"))

    def test_normalize_queue_controls_sanitizes_inputs(self):
        max_in_flight, drop_policy, input_fps_cap = normalize_queue_controls(
            None, " BAD ", "not-a-number"
        )
        self.assertEqual(max_in_flight, 64)
        self.assertEqual(drop_policy, "drop_oldest")
        self.assertIsNone(input_fps_cap)

    def test_normalize_queue_controls_latest_only_forces_single_inflight(self):
        max_in_flight, drop_policy, input_fps_cap = normalize_queue_controls(
            99, "latest_only", 30
        )
        self.assertEqual(max_in_flight, 1)
        self.assertEqual(drop_policy, "latest_only")
        self.assertEqual(input_fps_cap, 30.0)

    def test_should_pause_dispatch_applies_to_bounded_latency(self):
        self.assertFalse(should_pause_dispatch(in_flight=5, max_in_flight=6))
        self.assertTrue(should_pause_dispatch(in_flight=6, max_in_flight=6))

    def test_build_video_output_paths_handles_streams_and_files(self):
        filename, save_path, json_path = build_video_output_paths(
            "demo.mp4", "out", is_stream=False, save=True, save_json=True
        )
        self.assertEqual(filename, "demo.mp4")
        self.assertEqual(save_path, "out/demo.mp4")
        self.assertEqual(json_path, "out/demo.json")

        filename, save_path, json_path = build_video_output_paths(
            "0", "out", is_stream=True, save=False, save_json=True
        )
        self.assertEqual(filename, "stream_output.mp4")
        self.assertIsNone(save_path)
        self.assertEqual(json_path, "out/stream_output.json")

    def test_serialize_frame_detections_uses_public_json_shape(self):
        payload = serialize_frame_detections(7, [[1, 2, 3, 4, 0.9, 5]])
        self.assertEqual(
            payload,
            {
                "frame_idx": 7,
                "detections": [{"bbox": [1.0, 2.0, 3.0, 4.0], "conf": 0.9, "class": 5}],
            },
        )


if __name__ == "__main__":
    unittest.main()
