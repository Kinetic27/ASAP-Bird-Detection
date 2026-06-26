from __future__ import annotations

import glob
import os

from src.utils.resize import letterbox


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_image_paths(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.*")))
        return [path for path in files if path.lower().endswith(IMAGE_EXTENSIONS)]
    return [input_path]


def prepare_image_for_buffer(image, buffer_height: int, buffer_width: int, resize=None):
    image_height, image_width = image.shape[:2]
    ratio = (1.0, 1.0)
    pad = (0.0, 0.0)

    if resize or (image_height, image_width) != (buffer_height, buffer_width):
        image, ratio, pad = letterbox(
            image,
            new_shape=(buffer_height, buffer_width),
            auto=False,
            scaleup=True,
        )

    return image, ratio, pad, image_height, image_width


def restore_original_coordinates(boxes, ratio, pad, image_width: int, image_height: int):
    final_boxes = []
    pad_w, pad_h = pad
    ratio_x, ratio_y = ratio

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1 = (x1 - pad_w) / ratio_x
        y1 = (y1 - pad_h) / ratio_y
        x2 = (x2 - pad_w) / ratio_x
        y2 = (y2 - pad_h) / ratio_y

        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        final_boxes.append([x1, y1, x2, y2, conf, cls_id])

    return final_boxes
