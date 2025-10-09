from typing import Any, Type
from dataclass_wizard import fromdict

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.kernels.attention.attention_config import (
    AttentionConfigBMNK,
    AttentionConfigBSHD,
    AttentionConfigExtend,
)
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

BENCHMARKS: dict[str, dict[str, Type[KernelBenchmark]]] = {}
BENCHMARKS.update(GEMM_BENCH)
BENCHMARKS.update(ATTENTION_BENCH)
BENCHMARKS.update(CONV_BENCH)

LOAD_PROBLEMS = {
    "gemm": get_default_gemm_configs,
    "attention": get_default_attention_configs,
    "extend_attention": get_default_attention_configs,
    "conv": get_default_conv_configs,
}

CONFIG_CLASSES = {
    "gemm": GemmConfig,
    "attention": AttentionConfigBMNK,
    "extend_attention": AttentionConfigExtend,
    "conv": ConvConfig,
}


def create_benchmark(
    kernel_type: str,
    backend: str,
    kwargs: dict[str, Any],
    serialize=True,
) -> KernelBenchmark:
    BenchType = BENCHMARKS[kernel_type][backend]
    if serialize:
        return fromdict(BenchType, kwargs)
    else:
        return BenchType(**kwargs)
