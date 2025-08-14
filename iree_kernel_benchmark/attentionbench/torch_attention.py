from typing import Optional, override
import torch

from ..utils import *
from .attention_config import (
    AttentionAttributes,
    AttentionConfigBMNK,
    bmnk1k2_to_attention_attributes,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_ones,
)
import time


class TorchAttentionBenchmark(KernelBenchmark):
    def _clear_mem(self, *tensors):
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @override
    def bench_kernel(
        self, config: AttentionConfigBMNK, vmfb_filename, num_iterations=3, debug=False
    ):

        shape = bmnk1k2_to_attention_attributes(config)

        q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
        k_shape = (shape.num_query_heads, shape.kv_seq_len, shape.head_size)
        v_shape = (shape.num_query_heads, shape.kv_seq_len, shape.head_size_kv)

        q = device_randn(q_shape, dtype=dtype_to_torch(config.dtype))
        k = device_randn(k_shape, dtype=dtype_to_torch(config.dtype))
        v = device_randn(v_shape, dtype=dtype_to_torch(config.dtype))

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
            print(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return 0, False
        self._clear_mem(q, k, v)

        delta_time_ms = start_event.elapsed_time(end_event)
        delta_time_us = delta_time_ms * 1e3
        mean_time_us = delta_time_us / num_iterations

        return mean_time_us, True
