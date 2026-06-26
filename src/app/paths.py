from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_VIDEO = PROJECT_ROOT / "data" / "samples" / "DSC_1132_long.mp4"
DEFAULT_SAMPLE_IMAGE = PROJECT_ROOT / "data" / "samples" / "DSC_0024.JPG"


def sample_video_path() -> str:
    return os.environ.get("SAMPLE_VIDEO", str(DEFAULT_SAMPLE_VIDEO))


def sample_image_path() -> str:
    return os.environ.get("SAMPLE_IMAGE", str(DEFAULT_SAMPLE_IMAGE))
