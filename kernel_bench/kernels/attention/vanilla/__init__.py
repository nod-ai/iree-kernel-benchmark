from .backends.wave_vanilla_attention import WaveVanillaAttentionBenchmark
from .backends.iree_vanilla_attention import IREEVanillaAttentionBenchmark
from .backends.torch_vanilla_attention import TorchVanillaAttentionBenchmark
from .problems import get_vanilla_attention_configs


def get_default_attention_configs(kernel_type: str, backend_name: str):
    return [
        (tag, config) for tag, config in get_vanilla_attention_configs(use_fp8=True)
    ]


__all__ = [
    "WaveVanillaAttentionBenchmark",
    "IREEVanillaAttentionBenchmark",
    "TorchVanillaAttentionBenchmark",
    "get_default_attention_configs",
]
