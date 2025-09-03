from typing import List, Tuple
from wave_lang.kernel.wave.constraints import MMAType

from .gemm_utils import GemmTuningSpec
from .iree_gemm import IREEGemmBenchmark
from .wave_gemm import WaveGemmBenchmark
from .torch_gemm import TorchGemmBenchmark
from .hipblaslt_gemm import HipBLASLtGemmBenchmark
from .problems import (
    get_gemm_comparison,
    get_gemm_configs,
    get_tk_gemm_configs,
    get_b200_gemm_configs,
)


def get_default_gemm_configs(kernel_type: str, backend_name: str):
    return get_b200_gemm_configs(backend_name)
    if backend_name == "iree":
        configs = get_gemm_configs("f16", backend_name, False)
    else:
        all_configs = get_gemm_configs("f16", backend_name, False)
        configs = [
            (tag, config)
            for tag, config in all_configs
            if config.tA + config.tB == "NT"
        ]
    return configs


GEMM_BENCH = {
    "gemm": {
        "wave": WaveGemmBenchmark,
        "iree": IREEGemmBenchmark,
        "torch": TorchGemmBenchmark,
        "hipblaslt": HipBLASLtGemmBenchmark,
    }
}
