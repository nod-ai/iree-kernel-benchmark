from .bshd import *
from .extend import *
from .vanilla import *

ATTENTION_BENCH = {
    "attention": {
        "wave": WaveVanillaAttentionBenchmark,
        "iree": IREEVanillaAttentionBenchmark,
        "torch": TorchVanillaAttentionBenchmark,
        "triton": TritonVanillaAttentionBenchmark,
    },
    "bshd_attention": {
        "wave": WaveBSHDAttentionBenchmark,
        "triton": TritonBSHDAttentionBenchmark,
        "torch": TorchBSHDAttentionBenchmark,
    },
    "extend_attention": {
        "wave": WaveExtendAttentionBenchmark,
        "triton": TritonExtendAttentionBenchmark,
        "torch": TorchExtendAttentionBenchmark,
    },
}
