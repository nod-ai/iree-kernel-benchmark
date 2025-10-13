import torch
from aiter.ops.triton.prefill_attention import context_attention_fwd

from kernel_bench.config.types.attention import AttentionConfigBMNK
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.dtypes.device_context import DeviceContext
from kernel_bench.utils.torch_utils import benchmark_function_torch
from wave_lang.kernel.wave.utils.torch_utils import (
    device_ones,
    device_randn,
    device_zeros,
)


def get_triton_bmnk_inputs(config: AttentionConfigBMNK, device_ctx: DeviceContext):
    q_shape = (config.M, config.B, config.K1)
    k_shape = (config.K2, config.B, config.K1)
    v_shape = (config.K2, config.B, config.N)
    o_shape = (config.M, config.B, config.N)

    dtype = device_ctx.dtype_to_torch(config.dtype)
    q = device_randn(q_shape, dtype=dtype)
    k = device_randn(k_shape, dtype=dtype)
    v = device_randn(v_shape, dtype=dtype)
    o = device_randn(o_shape, dtype=dtype)

    b_start_loc = device_zeros((1,), dtype=torch.int32)
    b_seq_len = device_ones((1,), dtype=torch.int32) * config.M

    return q, k, v, o, b_start_loc, b_seq_len


class TritonVanillaAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBMNK

    def run_bench(self, device, num_iterations, timeout=None):
        q, k, v, o, b_start_loc, b_seq_len = get_triton_bmnk_inputs(
            self.config, self.device_ctx
        )

        try:
            mean_time_us = benchmark_function_torch(
                context_attention_fwd,
                iterations=50,
                compile=False,
                # Extend attention inputs
                q=q,
                k=k,
                v=v,
                o=o,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                max_input_len=self.config.M,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
