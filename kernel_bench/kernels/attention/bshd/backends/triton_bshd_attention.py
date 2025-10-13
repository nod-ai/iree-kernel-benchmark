import torch
from aiter.ops.triton.prefill_attention import context_attention_fwd

from kernel_bench.config.types.attention import AttentionConfigBSHD
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.dtypes.device_context import DeviceContext
from kernel_bench.utils.torch_utils import benchmark_function_torch
from wave_lang.kernel.wave.utils.torch_utils import (
    device_ones,
    device_randn,
    device_zeros,
)


def get_triton_bshd_inputs(config: AttentionConfigBSHD, device_ctx: DeviceContext):
    q_shape = (config.B * config.N_Q, config.H, config.D_Q)
    k_shape = (config.B * config.N_KV, config.H_KV, config.D_Q)
    v_shape = (config.B * config.N_KV, config.H_KV, config.D_KV)
    o_shape = (config.B * config.N_Q, config.H, config.D_KV)

    dtype = device_ctx.dtype_to_torch(config.dtype)
    q = device_randn(q_shape, dtype=dtype)
    k = device_randn(k_shape, dtype=dtype)
    v = device_randn(v_shape, dtype=dtype)
    o = device_randn(o_shape, dtype=dtype)

    b_start_loc = device_zeros((config.B,), dtype=torch.int32)
    b_seq_len = device_ones((config.B,), dtype=torch.int32) * config.N_Q

    return q, k, v, o, b_start_loc, b_seq_len


class TritonBSHDAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBSHD

    def run_bench(self, device, num_iterations, timeout=None):
        q, k, v, o, b_start_loc, b_seq_len = get_triton_bshd_inputs(
            self.config, self.device_ctx
        )

        try:
            mean_time_us = benchmark_function_torch(
                context_attention_fwd,
                iterations=50,
                compile=True,
                # Extend attention inputs
                q=q,
                k=k,
                v=v,
                o=o,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                max_input_len=self.config.N_Q,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
