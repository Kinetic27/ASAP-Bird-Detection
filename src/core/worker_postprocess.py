from __future__ import annotations

import json

import cv2
import numpy as np


def scale_global_boxes(boxes_data: np.ndarray, width: int, height: int, global_size: int) -> None:
    scale_x = width / global_size
    scale_y = height / global_size
    boxes_data[:, 0] *= scale_x
    boxes_data[:, 1] *= scale_y
    boxes_data[:, 2] *= scale_x
    boxes_data[:, 3] *= scale_y


def offset_patch_boxes(boxes_data: np.ndarray, offset_x: int, offset_y: int) -> None:
    boxes_data[:, 0] += offset_x
    boxes_data[:, 1] += offset_y
    boxes_data[:, 2] += offset_x
    boxes_data[:, 3] += offset_y


def append_boxes(final_boxes_list: list[list[float]], boxes_data: np.ndarray) -> None:
    for row in boxes_data:
        final_boxes_list.append(
            [
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                int(row[5]),
            ]
        )


def apply_nms(
    final_boxes_list: list[list[float]],
    score_threshold: float,
    nms_threshold: float = 0.5,
) -> list[list[float]]:
    if not final_boxes_list:
        return []

    all_boxes = np.array(final_boxes_list)
    boxes_xywh = all_boxes[:, :4].copy()
    boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
    boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
    scores = all_boxes[:, 4]
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.tolist(),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    if len(indices) == 0:
        return []
    return [final_boxes_list[index] for index in indices.flatten()]


def format_worker_output(
    frame_idx: int,
    total_duration: float,
    inference_duration: float,
    nms_duration: float,
    final_boxes: list[list[float]],
) -> str:
    return (
        f"{frame_idx}|{total_duration:.6f}|{inference_duration:.6f}|"
        f"{nms_duration:.6f}|{json.dumps(final_boxes)}"
    )
