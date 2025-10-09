from typing import Callable, Optional
import torch


def benchmark_function_torch(
    fn: Callable[..., None], *inputs, warmup=10, iterations=10, **kwinputs
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: could not benchmark torch function")

    try:
        # compiled_fn = torch.compile(fn)
        compiled_fn = fn
    except Exception as e:
        raise RuntimeError(f"Failed to compile torch kernel: {e}")

    try:
        torch.cuda.empty_cache()
        for _ in range(warmup):
            compiled_fn(*inputs, **kwinputs)
    except Exception as e:
        torch.cuda.synchronize()
        raise RuntimeError(f"Failed torch warmup runs: {e}")

    try:
        torch.cuda.empty_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for _ in range(iterations):
            compiled_fn(*inputs, **kwinputs)

        end_event.record()
        torch.cuda.synchronize()

    except Exception as e:
        raise e

    delta_time_ms = start_event.elapsed_time(end_event)
    delta_time_us = delta_time_ms * 1e3
    mean_time_us = delta_time_us / iterations
    return mean_time_us
