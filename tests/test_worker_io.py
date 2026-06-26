import io
import unittest

from src.core.worker_io import dispatch_to_first_available_worker


class _FakeProc:
    def __init__(self, *, alive=True, fail=False):
        self._alive = alive
        self._fail = fail
        self.stdin = self
        self.buffer = io.StringIO()

    def poll(self):
        return None if self._alive else 1

    def write(self, text):
        if self._fail:
            raise BrokenPipeError("simulated")
        self.buffer.write(text)

    def flush(self):
        return None


class WorkerIoTest(unittest.TestCase):
    def test_dispatch_skips_dead_and_failing_workers(self):
        dead = _FakeProc(alive=False)
        failing = _FakeProc(fail=True)
        healthy = _FakeProc()

        dispatch_to_first_available_worker([dead, failing, healthy], 7)

        self.assertEqual(healthy.buffer.getvalue(), "7\n")


if __name__ == "__main__":
    unittest.main()
