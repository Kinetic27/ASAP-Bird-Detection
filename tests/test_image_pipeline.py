import tempfile
import unittest
import importlib.util

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("optional dependency numpy is not installed")

from pathlib import Path

import numpy as np

from src.core.image_pipeline import (
    collect_image_paths,
    prepare_image_for_buffer,
    restore_original_coordinates,
)


class ImagePipelineTest(unittest.TestCase):
    def test_collect_image_paths_filters_non_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "a.jpg").write_text("x", encoding="utf-8")
            (tmp / "b.png").write_text("x", encoding="utf-8")
            (tmp / "ignore.txt").write_text("x", encoding="utf-8")

            images = collect_image_paths(tmp_dir)

        self.assertEqual(len(images), 2)
        self.assertTrue(images[0].endswith("a.jpg"))
        self.assertTrue(images[1].endswith("b.png"))

    def test_prepare_image_for_buffer_keeps_identity_when_shape_matches(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)

        prepared, ratio, pad, height, width = prepare_image_for_buffer(
            image, buffer_height=10, buffer_width=20
        )

        self.assertEqual(prepared.shape, image.shape)
        self.assertEqual(ratio, (1.0, 1.0))
        self.assertEqual(pad, (0.0, 0.0))
        self.assertEqual((height, width), (10, 20))

    def test_restore_original_coordinates_unpads_and_clamps(self):
        boxes = [[15.0, 10.0, 55.0, 50.0, 0.9, 1]]

        restored = restore_original_coordinates(
            boxes,
            ratio=(2.0, 2.0),
            pad=(5.0, 0.0),
            image_width=30,
            image_height=20,
        )

        self.assertEqual(restored, [[5.0, 5.0, 25.0, 20, 0.9, 1]])


if __name__ == "__main__":
    unittest.main()
