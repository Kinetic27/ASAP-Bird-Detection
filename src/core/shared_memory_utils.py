from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

try:
    from multiprocessing import resource_tracker
except ImportError:  # pragma: no cover
    resource_tracker = None


@dataclass
class SharedBufferBundle:
    shm: SharedMemory
    array: np.ndarray
    tensor: torch.Tensor
    info: dict


def create_shared_buffer(shape, dtype=np.uint8) -> SharedBufferBundle:
    total_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    shm = SharedMemory(create=True, size=total_bytes)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    tensor = torch.from_numpy(array)
    info = {"name": shm.name, "shape": tuple(shape), "dtype": dtype}
    return SharedBufferBundle(shm=shm, array=array, tensor=tensor, info=info)


def cleanup_shared_memory(shm: SharedMemory | None) -> None:
    """Close/unlink a shared-memory segment."""
    if shm is None:
        return

    try:
        shm.close()
    except Exception:
        pass

    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def untrack_shared_memory(shm: SharedMemory | None) -> None:
    """Detach a shared-memory handle from the current process resource tracker."""
    if shm is None or resource_tracker is None:
        return

    tracker_name = getattr(shm, "_name", None) or shm.name
    try:
        resource_tracker.unregister(tracker_name, "shared_memory")
    except Exception:
        pass
