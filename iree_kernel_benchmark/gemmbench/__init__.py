from typing import List, Tuple
from wave_lang.kernel.wave.constraints import MMAType

from ..utils import TuningConstraint
from .gemm_utils import GemmTuningSpec
from .iree_gemm import IREEGemmBenchmark
from .wave_gemm import WaveGemmBenchmark
from .torch_gemm import TorchGemmBenchmark
from .problems import get_gemm_configs, get_tk_gemm_configs


def get_default_gemm_configs(kernel_type: str, backend_name: str):
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


def get_gemm_tuning(kernel_type: str, backend_name: str):
    mfma_configs: List[MMAType] = [
        MMAType.F32_16x16x16_F16,
    ]
    tiling_constraints: List[TuningConstraint] = [
        TuningConstraint(name="BLOCK_M", min=16, max=256, step=8),
        TuningConstraint(name="BLOCK_N", min=16, max=256, step=8),
        TuningConstraint(name="BLOCK_K", min=16, max=128, step=4),
        TuningConstraint(name="ELEMS_PER_THREAD", min=4, max=4, step=1),
    ]
    return tiling_constraints, mfma_configs


GEMM_BENCH = {
    "gemm": {
        "wave": WaveGemmBenchmark,
        "iree": IREEGemmBenchmark,
        "torch": TorchGemmBenchmark,
    }
}
