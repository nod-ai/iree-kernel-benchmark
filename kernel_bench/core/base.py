from typing import Type

from kernel_bench.kernels.attention.attention_config import AttentionConfigBMNK
from kernel_bench.kernels.conv.conv_utils import ConvConfig
from kernel_bench.kernels.gemm.gemm_utils import GemmConfig

from kernel_bench.kernels.gemm import (
    GEMM_BENCH,
    get_default_gemm_configs,
)
from kernel_bench.kernels.attention import (
    ATTENTION_BENCH,
    get_default_attention_configs,
)
from kernel_bench.kernels.conv import (
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
