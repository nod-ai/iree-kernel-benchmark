from .backends.wave_attention import (
    WaveAttentionGQABenchmark,
    WaveAttentionMHABenchmark,
)
from .backends.iree_attention import IREEAttentionBenchmark
from .backends.torch_attention import TorchAttentionBenchmark
from .problems import get_attention_configs, get_attention_configs_gqa


def get_default_attention_configs(kernel_type: str, backend_name: str):
    if backend_name == "wavegqa":
        configs = [
            (tag, config.to_bshd()) for tag, config in get_attention_configs_gqa()
        ]
    elif backend_name in ["iree"]:
        configs = [
            (tag, config.to_bmnk1k2())
            for tag, config in get_attention_configs(use_fp8=True)
        ]
    else:
        configs = [
            (tag, config.to_bmnk1k2())
            for tag, config in get_attention_configs(use_fp8=False)
        ]
    return configs


ATTENTION_BENCH = {
    "attention": {
        "torch": TorchAttentionBenchmark,
        "wave": WaveAttentionMHABenchmark,
        "iree": IREEAttentionBenchmark,
        "wavegqa": WaveAttentionGQABenchmark,
    }
}
