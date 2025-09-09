from typing import override
import torch

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.device_utils import dtype_to_torch
from ..gemm_utils import GemmConfig


class TorchGemmBenchmark(KernelBenchmark):
    config: GemmConfig

    def _clear_mem(self, *tensors):
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @override
    def run_bench(self, device, num_iterations=1):
        config = self.config

        transposeA = config.tA == "T"
        transposeB = config.tB == "T"

        shape_a = (config.K, config.M) if transposeA else (config.M, config.K)
        shape_b = (config.N, config.K) if transposeB else (config.K, config.N)

        a_base = torch.rand(
            shape_a, dtype=dtype_to_torch(config.operand_element_type), device=device
        )
        b_base = torch.rand(
            shape_b, dtype=dtype_to_torch(config.operand_element_type), device=device
        )

        self._clear_mem()
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iterations):
                a = a_base.transpose(-2, -1) if transposeA else a_base
                b = b_base.transpose(-2, -1) if transposeB else b_base
                torch.matmul(a, b)
            end_event.record()
            torch.cuda.synchronize()

        except Exception as e:
            print(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return 0, False
        self._clear_mem(a_base, b_base)

        delta_time_ms = start_event.elapsed_time(end_event)
        delta_time_us = delta_time_ms * 1e3
        mean_time_us = delta_time_us / num_iterations

        return mean_time_us, True
