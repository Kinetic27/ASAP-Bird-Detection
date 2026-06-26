import queue
import time
import unittest
import importlib.util

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("optional dependency numpy is not installed")
if importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("optional dependency cv2 is not installed")
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional dependency torch is not installed")

from unittest import mock

import numpy as np

from src.core.loader import ThreadedStreamLoader


class FakeVideoCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.width = frames[0].shape[1]
        self.height = frames[0].shape[0]
        self.fps = 30.0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class LoaderEofDrainTest(unittest.TestCase):
    def test_loader_drains_pending_frames_after_reader_reaches_eof(self):
        import cv2

        frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(12)]
        fake_cap = FakeVideoCapture(frames)
        real_resize = cv2.resize

        class FakeSharedTensor:
            def __init__(self, array):
                self.array = array

            def share_memory_(self):
                return self.array

        def slow_resize(frame, target_size, interpolation=cv2.INTER_LINEAR):
            time.sleep(0.01)
            return real_resize(frame, target_size, interpolation=interpolation)

        with mock.patch("src.core.loader.cv2.VideoCapture", return_value=fake_cap), mock.patch(
            "src.core.loader.cv2.resize", side_effect=slow_resize
        ), mock.patch(
            "src.core.loader.torch.from_numpy",
            side_effect=lambda array: FakeSharedTensor(array),
        ):
            loader = ThreadedStreamLoader(
                "fake.mp4",
                patch_size=4,
                min_overlap=0,
                target_size=(2, 2),
                queue_size=2,
                drop_if_full=False,
            ).start()

            received = []
            deadline = time.time() + 5.0
            try:
                while time.time() < deadline and (loader.more() or len(received) < 12):
                    try:
                        idx, _frame_ref, _offsets = loader.q.get(timeout=0.05)
                        received.append(idx)
                    except queue.Empty:
                        continue
            finally:
                loader.stop()

        self.assertEqual(sorted(received), list(range(12)))


if __name__ == "__main__":
    unittest.main()
