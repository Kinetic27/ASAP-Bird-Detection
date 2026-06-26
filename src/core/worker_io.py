from __future__ import annotations

import time


def dispatch_to_first_available_worker(workers, frame_index: int, *, sleep_fn=time.sleep) -> None:
    pushed = False
    while not pushed:
        for proc in workers:
            if hasattr(proc, "poll") and proc.poll() is not None:
                continue
            try:
                proc.stdin.write(f"{frame_index}\n")
                proc.stdin.flush()
                pushed = True
                break
            except Exception:
                continue
        if not pushed:
            sleep_fn(0.001)
