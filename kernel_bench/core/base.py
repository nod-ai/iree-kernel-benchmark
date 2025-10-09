from typing import Any, Type
from dataclass_wizard import fromdict

from kernel_bench.config import get_global_registry
from kernel_bench.config.types import (
    GemmConfig,
    ConvConfig,
    AttentionConfigBSHD,
    AttentionConfigExtend,
    AttentionConfigBMNK,
)
from kernel_bench.core.template import KernelBenchmark


from kernel_bench.kernels.gemm import (
    GEMM_BENCH,
    get_default_gemm_configs,
)
from kernel_bench.kernels.conv import (
    CONV_BENCH,
    get_default_conv_configs,
)
from kernel_bench.kernels.attention import (
    ATTENTION_BENCH,
    get_default_attention_configs,
    get_default_bshd_attention_configs,
    get_default_extend_attention_configs,
)

BENCHMARKS: dict[str, dict[str, Type[KernelBenchmark]]] = {}
BENCHMARKS.update(GEMM_BENCH)
BENCHMARKS.update(ATTENTION_BENCH)
BENCHMARKS.update(CONV_BENCH)

LOAD_PROBLEMS = {
    "gemm": get_default_gemm_configs,
    "conv": get_default_conv_configs,
    "attention": get_default_attention_configs,
    "extend_attention": get_default_extend_attention_configs,
    "bshd_attention": get_default_bshd_attention_configs,
}

CONFIG_CLASSES = {
    "gemm": GemmConfig,
    "conv": ConvConfig,
    "attention": AttentionConfigBMNK,
    "extend_attention": AttentionConfigExtend,
    "bshd_attention": AttentionConfigBSHD,
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


# Initialize global config registry
_registry = get_global_registry()
_registry.register("gemm", GemmConfig, get_default_gemm_configs)
_registry.register("conv", ConvConfig, get_default_conv_configs)
_registry.register("attention", AttentionConfigBMNK, get_default_attention_configs)
_registry.register(
    "extend_attention", AttentionConfigExtend, get_default_extend_attention_configs
)
_registry.register(
    "bshd_attention", AttentionConfigBSHD, get_default_bshd_attention_configs
)
