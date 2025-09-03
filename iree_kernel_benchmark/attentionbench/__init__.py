from typing import List, Tuple
from wave_lang.kernel.wave.constraints import MMAType

from .wave_attention import WaveAttentionGQABenchmark, WaveAttentionMHABenchmark
from .iree_attention import IREEAttentionBenchmark
from .torch_attention import TorchAttentionBenchmark
from .problems import get_attention_configs, get_attention_configs_gqa


def get_default_attention_configs(kernel_type: str, backend_name: str):
    if backend_name == "wavegqa":
        configs = [
            (tag, config.to_bshd()) for tag, config in get_attention_configs_gqa()
        ]
    elif backend_name == "iree":
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
