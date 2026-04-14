"""
Profiling utilities for tracking memory usage and execution time.

Usage:
    from devtools.profiling import track_time, track_memory, profile

    # Context managers
    with track_time("Building AWSEM model"):
        model = frustratometer.AWSEM(structure)

    with track_memory("AWSEM sparse"):
        model = frustratometer.AWSEM(structure, sparse=True)

    # Decorator
    @profile
    def my_function():
        ...
"""

import time
import tracemalloc
import functools
from contextlib import contextmanager


@contextmanager
def track_time(label=""):
    """Context manager that prints elapsed wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    tag = f" [{label}]" if label else ""
    print(f"Time{tag}: {elapsed:.4f} s")


@contextmanager
def track_memory(label=""):
    """Context manager that prints peak memory allocated inside the block.

    Uses ``tracemalloc`` to measure only *new* Python allocations within the
    block, so the numbers reflect the incremental cost of the code inside.
    """
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    yield
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total = sum(s.size_diff for s in stats if s.size_diff > 0)
    tag = f" [{label}]" if label else ""
    print(f"Memory{tag}: {_fmt_bytes(total)}")


@contextmanager
def track_peak_memory(label=""):
    """Context manager that prints the peak memory usage (high-water mark)."""
    tracemalloc.start()
    yield
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tag = f" [{label}]" if label else ""
    print(f"Peak memory{tag}: {_fmt_bytes(peak)}")


def profile(func):
    """Decorator that prints time and peak memory for a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"{func.__name__}: {elapsed:.4f} s, peak {_fmt_bytes(peak)}")
        return result
    return wrapper


def _fmt_bytes(n):
    """Format byte count in human-readable units."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"
