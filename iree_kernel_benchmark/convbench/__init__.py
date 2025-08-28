from typing import List, Tuple
from wave_lang.kernel.wave.constraints import MMAType

from ..utils import TuningConstraint
from .conv_utils import ConvTuningSpec
from .wave_conv import WaveConvBenchmark
from .iree_conv import IREEConvBenchmark
from .torch_conv import TorchConvBenchmark
from .problems import get_conv_configs, get_tk_conv_configs


def get_default_conv_configs(kernel_type: str, backend_name: str):
    configs = get_tk_conv_configs()
    if backend_name == "torch":
        configs = [(tag, config) for tag, config in configs if "nchw" in config.OP]
    return configs


def get_conv_tuning(kernel_type: str, backend_name: str):
    mfma_configs: List[MMAType] = [
        MMAType.F32_16x16x16_F16,
    ]
    tiling_constraints: List[TuningConstraint] = [
        TuningConstraint(name="BLOCK_M", min=16, max=256, step=8),
        TuningConstraint(name="BLOCK_N", min=16, max=256, step=8),
        TuningConstraint(name="BLOCK_K", min=16, max=128, step=4),
        TuningConstraint(name="ELEMS_PER_THREAD", min=4, max=4, step=1),
    ]
    return ConvTuningSpec, tiling_constraints, mfma_configs


CONV_BENCH = {
    "conv": {
        "wave": WaveConvBenchmark,
        "iree": IREEConvBenchmark,
        "torch": TorchConvBenchmark,
    }
}
