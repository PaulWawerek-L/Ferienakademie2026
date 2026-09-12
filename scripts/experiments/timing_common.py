"""Timing harness shared by every project's inference benchmark.

Two things this exists to get right:

* **Warm-up.** The first call through a PyTorch graph pays for oneDNN algorithm
  selection, lazy allocator growth and page faults. On CPU that first run can be
  several times the steady-state cost, so timing it would measure setup, not
  inference.
* **What is excluded.** Reading the image, loading a checkpoint and building the
  model are not inference. Only the forward pass is inside the timed region.

Reports the median, not the mean: a single scheduler hiccup on a laptop skews a
mean and leaves the median untouched.
"""
import json
import os
import platform
import statistics
import time


def bench(fn, warmup=2, repeats=5, label=""):
    """Run fn() warmup+repeats times; return timing stats in seconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    stats = {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "repeats": repeats,
        "warmup": warmup,
        "samples_s": samples,
    }
    if label:
        print(f"  {label:<34} {stats['median_s']:8.3f} s  "
              f"(min {stats['min_s']:.3f}, max {stats['max_s']:.3f})")
    return stats


def env_info(threads=None):
    """Record what the numbers depend on, so they can be compared later."""
    info = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    if threads is not None:
        info["threads_requested"] = threads
    return info


def save(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}")
