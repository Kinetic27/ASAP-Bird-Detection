import unittest
import importlib.util

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("optional dependency numpy is not installed")
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional dependency torch is not installed")

from multiprocessing.shared_memory import SharedMemory

from src.core.shared_memory_utils import cleanup_shared_memory, create_shared_buffer


class SharedMemoryUtilsTest(unittest.TestCase):
    def test_create_shared_buffer_exposes_array_tensor_and_info(self):
        bundle = create_shared_buffer((2, 3, 4, 1))
        try:
            self.assertEqual(bundle.array.shape, (2, 3, 4, 1))
            self.assertEqual(tuple(bundle.tensor.shape), (2, 3, 4, 1))
            self.assertEqual(bundle.info["shape"], (2, 3, 4, 1))
            self.assertEqual(bundle.info["name"], bundle.shm.name)
        finally:
            cleanup_shared_memory(bundle.shm)

    def test_cleanup_shared_memory_unlinks_segment(self):
        shm = SharedMemory(create=True, size=8)
        name = shm.name

        cleanup_shared_memory(shm)

        attached = None
        try:
            attached = SharedMemory(name=name)
            self.fail("shared memory segment should have been unlinked")
        except FileNotFoundError:
            pass
        finally:
            if attached is not None:
                cleanup_shared_memory(attached)


if __name__ == "__main__":
    unittest.main()
