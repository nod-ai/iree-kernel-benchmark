import torch
from aiter.ops.triton.prefill_attention import (
    context_attention_fwd as triton_prefill_attention,
)

from kernel_bench.config.types.attention import AttentionConfigBMNK
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.kernels.attention.vanilla.data import create_bmnk_attention_inputs
from kernel_bench.utils.dtypes.device_context import DeviceContext
from kernel_bench.utils.torch_utils import benchmark_function_torch
from wave_lang.kernel.wave.utils.torch_utils import (
    device_ones,
    device_randn,
    device_zeros,
)


def get_triton_bmnk_inputs(config: AttentionConfigBMNK, device_ctx: DeviceContext):
    q, k, v, o = create_bmnk_attention_inputs(config, device_ctx)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    o = o.transpose(0, 1)

    b_start_loc = device_zeros((1,), dtype=torch.int32)
    b_seq_len = device_ones((1,), dtype=torch.int32) * config.M // 2

    return q, k, v, o, b_start_loc, b_seq_len


class TritonVanillaAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBMNK

    def run_bench(self, device, num_iterations, timeout=None):
        q, k, v, o, b_start_loc, b_seq_len = get_triton_bmnk_inputs(
            self.config, self.device_ctx
        )
        print(f"q={q.shape} k={k.shape} v={v.shape} o={o.shape}")

        triton_prefill_attention(q, k, v, o, b_start_loc, b_seq_len, self.config.M)
        torch.save(
            o.transpose(0, 1), f"results/outputs/triton/{self.config.get_name()}.pt"
        )

        try:
            mean_time_us = benchmark_function_torch(
                triton_prefill_attention,
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
                is_causal=False,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
