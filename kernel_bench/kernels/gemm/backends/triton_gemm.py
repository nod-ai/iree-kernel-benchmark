import torch
from kernel_bench.core.template import KernelBenchmark

from typing import Optional

from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a8w8 import gemm_a8w8

from kernel_bench.kernels.gemm.gemm_utils import GemmConfig
from kernel_bench.utils.device_utils import (
    dtype_max_value,
    dtype_to_bits,
    dtype_to_torch,
)
from kernel_bench.utils.torch_utils import benchmark_function_torch


class TritonGemmBenchmark(KernelBenchmark):
    config: GemmConfig

    def validate_config(self):
        input_dtype = self.config.dtype
        if input_dtype not in ["f16", "bf16"]:
            return False

        variant = self.config.tA + self.config.tB
        if variant != "NT":
            return False

        return True

    def run_bench(self, device, num_iterations=1, timeout=None):
        config = self.config

        in_dtype = config.dtype
        in_dtype_torch = self.device_context.get_bench_dtype(in_dtype).to_torch()
        out_dtype_torch = self.device_context.get_bench_dtype("f32").to_torch()

        variant = config.tA + config.tB
        if variant != "NT":
            self.logger.error(
                f"GEMM variant {variant} not supported for Triton backend"
            )

        x = torch.rand((config.M, config.K), dtype=in_dtype_torch, device="cuda")
        w = torch.rand((config.N, config.K), dtype=in_dtype_torch, device="cuda")
        y = torch.zeros((config.M, config.N), dtype=out_dtype_torch, device="cuda")

        try:
            if "f16" in in_dtype:
                mean_time_us = benchmark_function_torch(
                    gemm_a16w16,
                    x,
                    w,
                    in_dtype_torch,
                    y,
                    iterations=num_iterations,
                )
            elif "f8" in in_dtype:
                in_dtype_max = dtype_max_value(in_dtype_torch)

                max_x = x.abs().float().amax(dim=1, keepdim=True)
                x_scale = max_x / in_dtype_max
                x = x / x_scale
                x = x.to(in_dtype_torch)

                max_w = w.abs().float().amax(dim=1, keepdim=True).T.contiguous()
                w_scale = max_w / in_dtype_max
                w = w / w_scale.T
                w = w.to(in_dtype_torch)

                bias = (
                    torch.rand((1, config.N), dtype=torch.float32, device="cuda") * 10
                )

                mean_time_us = benchmark_function_torch(
                    gemm_a8w8,
                    x,
                    w,
                    x_scale,
                    w_scale,
                    bias,
                    in_dtype_torch,
                    y,
                    iterations=num_iterations,
                )

        except Exception as e:
            self.logger.error(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
