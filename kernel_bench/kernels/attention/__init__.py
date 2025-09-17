from .backends.wave_attention import (
    WaveAttentionGQABenchmark,
    WaveAttentionMHABenchmark,
    WaveExtendAttentionBenchmark,
)
from .backends.iree_attention import IREEAttentionBenchmark
from .backends.torch_attention import TorchAttentionBenchmark
from .problems import (
    get_attention_configs,
    get_attention_configs_gqa,
    get_extend_attention_configs,
)


def get_default_attention_configs(kernel_type: str, backend_name: str):
    if kernel_type == "attention":
        if backend_name == "wavegqa":
            return [
                (tag, config.to_bshd()) for tag, config in get_attention_configs_gqa()
            ]
        elif backend_name in ["iree"]:
            return [
                (tag, config.to_bmnk1k2())
                for tag, config in get_attention_configs(use_fp8=True)
            ]
        else:
            return [
                (tag, config.to_bmnk1k2())
                for tag, config in get_attention_configs(use_fp8=False)
            ]
    elif kernel_type == "extend_attention":
        return [
            (tag, config.to_bmnk1k2()) for tag, config in get_extend_attention_configs()
        ]


ATTENTION_BENCH = {
    "attention": {
        "torch": TorchAttentionBenchmark,
        "wave": WaveAttentionMHABenchmark,
        "iree": IREEAttentionBenchmark,
        # "wavegqa": WaveAttentionGQABenchmark,
    },
    "extend_attention": {
        "wave": WaveExtendAttentionBenchmark,
    },
}
