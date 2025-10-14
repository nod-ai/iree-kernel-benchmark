from .backends.triton_extend_attention import TritonExtendAttentionBenchmark
from .backends.torch_extend_attention import TorchExtendAttentionBenchmark
from .backends.wave_extend_attention import WaveExtendAttentionBenchmark
from .problems import get_extend_attention_configs


def get_default_extend_attention_configs(kernel_type: str, backend_name: str):
    return get_extend_attention_configs()


__all__ = [
    "TritonExtendAttentionBenchmark",
    "TorchExtendAttentionBenchmark",
    "WaveExtendAttentionBenchmark",
    "get_default_extend_attention_configs",
]
