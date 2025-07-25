from typing import Optional
import torch

from iree_kernel_benchmark.utils.wave_utils import DTYPE_TO_TORCH, dtype_to_torch
from .attention_config import AttentionAttributes
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_ones,
)
import time


def clear_mem(*tensors):
    for tensor in tensors:
        del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_torch_attention(
    config: AttentionAttributes, num_iterations: int = 3
) -> Optional[float]:

    q_shape = (config.num_query_heads, config.query_seq_len, config.head_size)
    k_shape = (config.num_query_heads, config.kv_seq_len, config.head_size)
    v_shape = (config.num_query_heads, config.kv_seq_len, config.head_size_kv)

    q = device_randn(q_shape, dtype=dtype_to_torch(config.dtype))
    k = device_randn(k_shape, dtype=dtype_to_torch(config.dtype))
    v = device_randn(v_shape, dtype=dtype_to_torch(config.dtype))

    clear_mem()
    try:
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        end_time = time.perf_counter()
    except Exception as e:
        print(f"Failed to benchmark kernel {config.get_name()}: {e}")
        return None
    clear_mem(q, k, v)

    delta_time_seconds = end_time - start_time
    delta_time_us = delta_time_seconds * 1e6
    mean_time_us = delta_time_us / num_iterations

    return mean_time_us
