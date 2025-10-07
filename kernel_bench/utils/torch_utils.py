from typing import Callable, Optional
import torch


def benchmark_function_torch(
    fn: Callable[..., None], *inputs, warmup=10, iterations=10, **kwinputs
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: could not benchmark torch function")

    try:
        torch.cuda.empty_cache()
        for _ in range(warmup):
            fn(*inputs, **kwinputs)
    except Exception as e:
        raise RuntimeError(f"Failed torch warmup runs: {e}")

    times_us = [0] * iterations

    try:
        torch.cuda.empty_cache()

        for i in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            fn(*inputs, **kwinputs)
            end_event.record()
            torch.cuda.synchronize()

            delta_time_ms = start_event.elapsed_time(end_event)
            delta_time_us = delta_time_ms * 1e3

            times_us[i] = delta_time_us

    except Exception as e:
        raise e

    mean_time_us = sum(times_us) / iterations
    return mean_time_us
