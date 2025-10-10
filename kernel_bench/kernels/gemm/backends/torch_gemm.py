from typing import override
import torch

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.torch_utils import benchmark_function_torch
from ..gemm_utils import GemmConfig


def run_torch_matmul(
    a_base: torch.Tensor, b_base: torch.Tensor, transposeA: bool, transposeB: bool
):
    a = a_base.transpose(-2, -1) if transposeA else a_base
    b = b_base.transpose(-2, -1) if transposeB else b_base
    c = torch.matmul(a, b)
    return c


class TorchGemmBenchmark(KernelBenchmark):
    config: GemmConfig

    @override
    def validate_config(self):
        if "f8" in self.config.dtype:
            return False
        return True

    @override
    def run_bench(self, device, num_iterations, timeout):
        config = self.config

        transposeA = config.tA == "T"
        transposeB = config.tB == "T"

        shape_a = (config.K, config.M) if transposeA else (config.M, config.K)
        shape_b = (config.N, config.K) if transposeB else (config.K, config.N)

        dtype = self.device_ctx.dtype_to_torch(config.dtype)
        a_base = torch.rand(shape_a, dtype=dtype, device="cuda")
        b_base = torch.rand(shape_b, dtype=dtype, device="cuda")

        try:
            mean_time_us = benchmark_function_torch(
                run_torch_matmul,
                a_base,
                b_base,
                transposeA,
                transposeB,
                iterations=100,
            )

        except Exception as e:
            self.logger.error(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
