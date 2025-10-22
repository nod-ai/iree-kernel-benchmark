import torch

from kernel_bench.config.types.attention import AttentionConfigBMNK
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.kernels.attention.bshd.backends.triton_bshd_attention import (
    triton_bshd_attention_forward,
)
from kernel_bench.kernels.attention.bshd.bshd_utils import get_bshd_inputs
from kernel_bench.utils.torch_utils import benchmark_function_torch


class TritonVanillaAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBMNK

    def run_bench(self, device, num_iterations, timeout=None):
        config = self.config
        in_dtype = self.device_ctx.dtype_to_torch(config.dtype)

        q, k, v, metadata = get_bshd_inputs(
            Z=1,
            HQ=config.B,
            HK=config.B,
            N_CTX_Q=config.M,
            N_CTX_K=config.K2,
            D_HEAD=config.N,
            dtype=in_dtype,
            layout="bhsd",
            requires_grad=False,
        )
        o = torch.empty_like(q)

        try:
            mean_time_us = benchmark_function_torch(
                triton_bshd_attention_forward,
                iterations=100,
                compile=False,
                # Extend attention inputs
                q=q,
                k=k,
                v=v,
                o=o,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
