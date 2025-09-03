from typing import Type
from .attentionbench.attention_config import AttentionConfigBMNK
from .convbench.conv_utils import ConvConfig
from .gemmbench.gemm_utils import GemmConfig

from .gemmbench import (
    GEMM_BENCH,
    get_default_gemm_configs,
)
from .attentionbench import (
    ATTENTION_BENCH,
    get_default_attention_configs,
)
from .convbench import (
    CONV_BENCH,
    get_default_conv_configs,
)

BENCHMARKS: dict[str, dict[str, Type]] = {}
BENCHMARKS.update(GEMM_BENCH)
BENCHMARKS.update(ATTENTION_BENCH)
BENCHMARKS.update(CONV_BENCH)

LOAD_PROBLEMS = {
    "gemm": get_default_gemm_configs,
    "attention": get_default_attention_configs,
    "conv": get_default_conv_configs,
}

CONFIG_CLASSES = {
    "gemm": GemmConfig,
    "attention": AttentionConfigBMNK,
    "conv": ConvConfig,
}
