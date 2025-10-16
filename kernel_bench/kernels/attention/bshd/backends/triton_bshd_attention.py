import os
import torch
from kernel_bench.config.types.attention import AttentionConfigBSHD
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.dtypes.device_context import DeviceContext
from kernel_bench.utils.torch_utils import benchmark_function_torch
from kernel_bench.utils.triton_utils import (
    get_triton_bshd_inputs,
    triton_bshd_attention_forward,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_ones,
    device_randn,
    device_zeros,
)


class TritonBSHDAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBSHD

    def run_bench(self, device, num_iterations, timeout=None):
        config = self.config
        in_dtype = self.device_ctx.dtype_to_torch(config.dtype)

        q, k, v, metadata = get_triton_bshd_inputs(
            Z=config.B,
            HQ=config.H,
            HK=config.H_KV,
            N_CTX_Q=config.N_Q,
            N_CTX_K=config.N_KV,
            D_HEAD=config.D_Q,
            dtype=in_dtype,
            layout="bshd",
            requires_grad=False,
        )
        o = torch.empty_like(q)

        # triton_bshd_attention_forward(q, k, v, o, metadata)
        # os.makedirs("results/inputs/bshd_attention/triton", exist_ok=True)
        # os.makedirs("results/outputs/bshd_attention/triton", exist_ok=True)
        # torch.save(q, f"results/inputs/bshd_attention/triton/{config.get_name()}_q.pt")
        # torch.save(k, f"results/inputs/bshd_attention/triton/{config.get_name()}_k.pt")
        # torch.save(v, f"results/inputs/bshd_attention/triton/{config.get_name()}_v.pt")
        # torch.save(o, f"results/outputs/bshd_attention/triton/{config.get_name()}.pt")

        try:
            mean_time_us = benchmark_function_torch(
                triton_bshd_attention_forward,
                warmup=20,
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
