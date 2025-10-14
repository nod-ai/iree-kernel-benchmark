import torch
from kernel_bench.config.types.attention import AttentionConfigBMNK
from kernel_bench.config.types.attention.vanilla_attention_config import (
    bmnk1k2_to_attention_attributes,
)
from kernel_bench.utils.dtypes.device_context import DeviceContext
from wave_lang.kernel.wave.utils.torch_utils import (
    device_empty,
    device_randn,
    device_zeros,
)


def create_bmnk_attention_inputs(
    config: AttentionConfigBMNK, device_ctx: DeviceContext
):
    torch.manual_seed(42)

    dtype = device_ctx.dtype_to_torch(config.dtype)

    q_shape = (config.B, config.M, config.K1)
    k_shape = (config.B, config.K2, config.K1)
    v_shape = (config.B, config.K2, config.N)
    o_shape = (config.B, config.M, config.N)

    q = device_randn(q_shape, dtype=dtype)
    k = device_randn(k_shape, dtype=dtype)
    v = device_randn(v_shape, dtype=dtype)
    o = device_empty(o_shape, dtype=dtype)

    return q, k, v, o
