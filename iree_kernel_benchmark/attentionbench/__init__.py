from typing import List, Tuple
from wave_lang.kernel.wave.constraints import MMAType

from ..utils import TuningConstraint
from .attention_utils import AttentionBMNKTuningSpec, AttentionBSHDTuningSpec
from .wave_attention import WaveAttentionGQABenchmark, WaveAttentionMHABenchmark
from .iree_attention import IREEAttentionBenchmark
from .torch_attention import TorchAttentionBenchmark
from .problems import get_attention_configs, get_attention_configs_gqa


def get_default_attention_configs(kernel_type: str, backend_name: str):
    if backend_name == "wavegqa":
        configs = [
            (tag, config.to_bshd()) for tag, config in get_attention_configs_gqa()
        ]
    elif backend_name == "iree":
        configs = [
            (tag, config.to_bmnk1k2())
            for tag, config in get_attention_configs(use_fp8=True)
        ]
    else:
        configs = [
            (tag, config.to_bmnk1k2())
            for tag, config in get_attention_configs(use_fp8=False)
        ]
    return configs


def get_attention_tuning(kernel_type: str, backend_name: str):
    if backend_name == "wavegqa":
        mfma_configs: List[Tuple[MMAType, MMAType]] = [
            (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
            (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
            (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
            (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        ]
        tiling_constraints: List[TuningConstraint] = [
            TuningConstraint(name="BLOCK_B", min=1, max=1, step=1),
            TuningConstraint(name="BLOCK_H", min=1, max=2, step=1),
            TuningConstraint(name="BLOCK_N_Q", min=16, max=128, step=16),
            TuningConstraint(name="BLOCK_D_KV", min=16, max=128, step=16),
            TuningConstraint(name="BLOCK_N_KV", min=16, max=64, step=16),
        ]
        tuning_spec_class = AttentionBSHDTuningSpec
    else:
        mfma_configs: List[Tuple[MMAType, MMAType]] = [
            (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
            (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
            (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
            (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        ]
        tiling_constraints: List[TuningConstraint] = [
            TuningConstraint(name="BLOCK_B", min=1, max=1, step=1),
            TuningConstraint(name="BLOCK_M", min=32, max=256, step=8),
            TuningConstraint(name="BLOCK_N", min=16, max=128, step=4),
            TuningConstraint(name="BLOCK_K2", min=32, max=256, step=8),
        ]
        tuning_spec_class = AttentionBMNKTuningSpec
    return tuning_spec_class, tiling_constraints, mfma_configs


ATTENTION_BENCH = {
    "attention": {
        "torch": TorchAttentionBenchmark,
        "wave": WaveAttentionMHABenchmark,
        "iree": IREEAttentionBenchmark,
        "wavegqa": WaveAttentionGQABenchmark,
    }
}
