from .bshd import *
from .extend import *
from .vanilla import *

ATTENTION_BENCH = {
    "attention": {
        "wave": WaveVanillaAttentionBenchmark,
        "iree": IREEVanillaAttentionBenchmark,
        "torch": TorchVanillaAttentionBenchmark,
    },
    "bshd_attention": {
        "wave": WaveBSHDAttentionBenchmark,
    },
    "extend_attention": {
        "wave": WaveExtendAttentionBenchmark,
        "triton": TritonExtendAttentionBenchmark,
    },
}
