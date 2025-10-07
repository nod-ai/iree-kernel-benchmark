from typing import List, Tuple
from wave_lang.kernel.wave.constraints import MMAType

from kernel_bench.kernels.gemm.backends.triton_gemm import TritonGemmBenchmark

from .backends.iree_gemm import IREEGemmBenchmark
from .backends.wave_gemm import WaveGemmBenchmark
from .backends.torch_gemm import TorchGemmBenchmark
from .backends.hipblaslt_gemm import HipBLASLtGemmBenchmark
from .problems import (
    get_gemm_comparison,
    get_gemm_configs,
    get_paper_gemms,
    get_tk_gemm_configs,
    get_b200_gemm_configs,
)


def get_default_gemm_configs(kernel_type: str, backend_name: str):
    configs = []
    # configs += get_b200_gemm_configs(backend_name)
    configs += get_gemm_configs("f16")
    configs += get_gemm_configs("bf16")
    configs += get_gemm_configs("f8")
    configs += get_paper_gemms()
    return configs


GEMM_BENCH = {
    "gemm": {
        "wave": WaveGemmBenchmark,
        "iree": IREEGemmBenchmark,
        "torch": TorchGemmBenchmark,
        "triton": TritonGemmBenchmark,
        "hipblaslt": HipBLASLtGemmBenchmark,
    }
}
