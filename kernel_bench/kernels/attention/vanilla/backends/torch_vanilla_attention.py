from typing import override
import torch

from kernel_bench.config.types.attention.vanilla_attention_config import (
    bmnk1k2_to_attention_attributes,
)
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.torch_utils import benchmark_function_torch
from kernel_bench.config.types.attention import AttentionConfigBMNK


class TorchVanillaAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBMNK

    @override
    def validate_config(self):
        if "f8" in self.config.dtype:
            return False
        return True

    @override
    def run_bench(self, device, num_iterations, timeout):
        config = self.config
        shape = bmnk1k2_to_attention_attributes(config)

        q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
        k_shape = (shape.num_query_heads, shape.kv_seq_len, shape.head_size)
        v_shape = (shape.num_query_heads, shape.kv_seq_len, shape.head_size_kv)

        dtype = self.device_ctx.dtype_to_torch(config.dtype)

        q = torch.randn(q_shape, dtype=dtype, device="cuda")
        k = torch.randn(k_shape, dtype=dtype, device="cuda")
        v = torch.randn(v_shape, dtype=dtype, device="cuda")

        try:
            mean_time_us = benchmark_function_torch(
                torch.nn.functional.scaled_dot_product_attention,
                q,
                k,
                v,
                attn_mask=None,
                iterations=50,
                compile=True,
            )

        except Exception as e:
            self.logger.error(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
