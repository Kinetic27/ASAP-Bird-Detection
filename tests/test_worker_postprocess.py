import unittest
import importlib.util

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("optional dependency numpy is not installed")
if importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("optional dependency cv2 is not installed")

from unittest import mock

import numpy as np

from src.core.worker_postprocess import (
    append_boxes,
    apply_nms,
    format_worker_output,
    offset_patch_boxes,
    scale_global_boxes,
)


class WorkerPostprocessTest(unittest.TestCase):
    def test_scale_global_boxes_scales_to_original_frame(self):
        boxes = np.array([[10.0, 20.0, 30.0, 40.0, 0.9, 1.0]])

        scale_global_boxes(boxes, width=1920, height=1080, global_size=960)

        np.testing.assert_allclose(
            boxes[:, :4],
            np.array([[20.0, 22.5, 60.0, 45.0]]),
        )

    def test_offset_patch_boxes_applies_patch_origin(self):
        boxes = np.array([[1.0, 2.0, 4.0, 5.0, 0.9, 0.0]])

        offset_patch_boxes(boxes, offset_x=100, offset_y=50)

        np.testing.assert_allclose(
            boxes[:, :4],
            np.array([[101.0, 52.0, 104.0, 55.0]]),
        )

    def test_append_boxes_converts_numpy_rows_to_serializable_lists(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0, 0.95, 2.0]])
        collected = []

        append_boxes(collected, boxes)

        self.assertEqual(collected, [[1.0, 2.0, 3.0, 4.0, 0.95, 2]])

    @mock.patch("src.core.worker_postprocess.cv2.dnn.NMSBoxes", return_value=np.array([[1]]))
    def test_apply_nms_returns_selected_boxes(self, mock_nms):
        boxes = [
            [0.0, 0.0, 10.0, 10.0, 0.5, 0],
            [1.0, 1.0, 9.0, 9.0, 0.9, 0],
        ]

        selected = apply_nms(boxes, score_threshold=0.2, nms_threshold=0.61)

        self.assertEqual(selected, [[1.0, 1.0, 9.0, 9.0, 0.9, 0]])
        self.assertEqual(mock_nms.call_args.kwargs["score_threshold"], 0.2)
        self.assertEqual(mock_nms.call_args.kwargs["nms_threshold"], 0.61)

    def test_format_worker_output_uses_protocol_shape(self):
        line = format_worker_output(3, 0.12, 0.08, 0.01, [[1, 2, 3, 4, 0.9, 0]])
        self.assertEqual(line, '3|0.120000|0.080000|0.010000|[[1, 2, 3, 4, 0.9, 0]]')


if __name__ == "__main__":
    unittest.main()
