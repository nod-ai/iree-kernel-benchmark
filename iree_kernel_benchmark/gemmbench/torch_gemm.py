from dataclasses import asdict
from typing import Optional, override
import torch

from iree_kernel_benchmark.utils.template import KernelBenchmark
from iree_kernel_benchmark.utils.wave_utils import DTYPE_TO_TORCH, dtype_to_torch
from .gemm_utils import GemmConfig
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_ones,
)
import time


class TorchGemmBenchmark(KernelBenchmark):
    def _clear_mem(self, *tensors):
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @override
    def bench_kernel(
        self, config: GemmConfig, vmfb_filename, num_iterations=3, debug=False
    ):
        transposeA = config.tA == "T"
        transposeB = config.tB == "T"

        shape_a = (config.K, config.M) if transposeA else (config.M, config.K)
        shape_b = (config.N, config.K) if transposeB else (config.K, config.N)

        a_base = device_randn(
            shape_a, dtype=dtype_to_torch(config.operand_element_type)
        )
        b_base = device_randn(
            shape_b, dtype=dtype_to_torch(config.operand_element_type)
        )

        self._clear_mem()
        try:
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                a = a_base.transpose(-2, -1) if transposeA else a_base
                b = b_base.transpose(-2, -1) if transposeB else b_base
                torch.matmul(a, b)
            end_time = time.perf_counter()
        except Exception as e:
            print(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return 0, False
        self._clear_mem(a_base, b_base)

        delta_time_seconds = end_time - start_time
        delta_time_us = delta_time_seconds * 1e6
        mean_time_us = delta_time_us / num_iterations

        return mean_time_us, True
