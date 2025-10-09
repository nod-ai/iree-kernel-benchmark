from .backends.wave_bshd_attention import WaveBSHDAttentionBenchmark
from .problems import cai_attn


def get_default_bshd_attention_configs(kernel_type: str, backend_name: str):
    return [("cai", config) for config in cai_attn("bf16")]


__all__ = [
    "WaveBSHDAttentionBenchmark",
    "get_default_bshd_attention_configs",
]
