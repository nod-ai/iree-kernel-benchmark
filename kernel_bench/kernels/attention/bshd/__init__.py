from .backends.torch_bshd_attention import TorchBSHDAttentionBenchmark
from .backends.triton_bshd_attention import TritonBSHDAttentionBenchmark
from .backends.wave_bshd_attention import WaveBSHDAttentionBenchmark
from .problems import get_bshd_attention_configs


def get_default_bshd_attention_configs(kernel_type: str, backend_name: str):
    return get_bshd_attention_configs()


__all__ = [
    "TorchBSHDAttentionBenchmark",
    "TritonBSHDAttentionBenchmark",
    "WaveBSHDAttentionBenchmark",
    "get_default_bshd_attention_configs",
]
