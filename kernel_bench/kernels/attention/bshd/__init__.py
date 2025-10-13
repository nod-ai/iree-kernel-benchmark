from .backends.torch_bshd_attention import TorchBSHDAttentionBenchmark
from .backends.triton_bshd_attention import TritonBSHDAttentionBenchmark
from .backends.wave_bshd_attention import WaveBSHDAttentionBenchmark
from .problems import cai_attn


def get_default_bshd_attention_configs(kernel_type: str, backend_name: str):
    return [("cai", config) for config in cai_attn()]


__all__ = [
    "TorchBSHDAttentionBenchmark",
    "TritonBSHDAttentionBenchmark",
    "WaveBSHDAttentionBenchmark",
    "get_default_bshd_attention_configs",
]
