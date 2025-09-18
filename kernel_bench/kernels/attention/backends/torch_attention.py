from typing import override
import torch

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.device_utils import dtype_to_torch
from ..attention_config import AttentionConfigBMNK


class TorchAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigBMNK

    def _clear_mem(self, *tensors):
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @override
    def run_bench(self, device, num_iterations=1):
        config = self.config
        shape = config.attributes

        q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
        k_shape = (shape.num_query_heads, shape.kv_seq_len, shape.head_size)
        v_shape = (shape.num_query_heads, shape.kv_seq_len, shape.head_size_kv)

        q = torch.randn(q_shape, dtype=dtype_to_torch(config.dtype), device=device)
        k = torch.randn(k_shape, dtype=dtype_to_torch(config.dtype), device=device)
        v = torch.randn(v_shape, dtype=dtype_to_torch(config.dtype), device=device)

        self._clear_mem()
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iterations):
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None
                )
            end_event.record()
            torch.cuda.synchronize()

        except Exception as e:
            self.logger.error(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return 0, False
        self._clear_mem(q, k, v)

        delta_time_ms = start_event.elapsed_time(end_event)
        delta_time_us = delta_time_ms * 1e3
        mean_time_us = delta_time_us / num_iterations

        return mean_time_us, True
