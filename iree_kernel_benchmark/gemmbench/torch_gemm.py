from dataclasses import asdict
from typing import Optional
import torch

from iree_kernel_benchmark.utils.wave_utils import DTYPE_TO_TORCH
from .gemm_utils import GemmConfig
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_ones,
)
import time


def benchmark_torch_gemm(
    config: GemmConfig, num_iterations: int = 3
) -> Optional[float]:
    shape_a = (config.K, config.M) if config.tA == "T" else (config.M, config.K)
    shape_b = (config.N, config.K) if config.tB == "T" else (config.K, config.N)

    a_base = device_randn(shape_a, dtype=DTYPE_TO_TORCH[config.operand_element_type])
    b_base = device_randn(shape_b, dtype=DTYPE_TO_TORCH[config.operand_element_type])

    try:
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            a = a_base.transpose(-2, -1) if config.tA == "T" else a_base
            b = b_base.transpose(-2, -1) if config.tB == "T" else b_base
            torch.matmul(a, b)
        end_time = time.perf_counter()
    except Exception as e:
        print(f"Failed to benchmark kernel {config.get_name()}: {e}")
        return None

    delta_time_seconds = end_time - start_time
    delta_time_us = delta_time_seconds * 1e6
    mean_time_us = delta_time_us / num_iterations

    return mean_time_us
