from kernel_bench.tuning.hyperparam import (
    TuningParameter,
    CategoricalBounds,
    IntegerBounds,
)
from kernel_bench.core.template import WaveKernel, WaveKernelBenchmark
from kernel_bench.utils.device_utils import dtype_to_torch
from wave_lang.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from ..attention_config import (
    AttentionConfigBMNK,
    AttentionConfigBSHD,
    bmnk1k2_to_attention_attributes,
    bshd_to_attention_attributes,
)
from typing import override

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.kernel.wave.templates.gqa_vanilla_attention import (
    get_gqa_bshd_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


class WaveAttentionMHABenchmark(WaveKernelBenchmark):
    config: AttentionConfigBMNK

    def __post_init__(self):
        super().__post_init__()

        if "f8" in self.config.dtype:
            mfma_bounds = CategoricalBounds(
                [
                    (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
                    (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
                ]
            )
        else:
            mfma_bounds = CategoricalBounds(
                [
                    (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                    (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
                ]
            )

        self.mfma_variant = TuningParameter(
            "MFMA_VARIANT", mfma_bounds, initial_value=0, include_hyperparam=False
        )
        self.BLOCK_B = TuningParameter("BLOCK_B", IntegerBounds(min=1, max=1, step=1))
        self.BLOCK_H = TuningParameter("BLOCK_H", IntegerBounds(min=1, max=2, step=1))
        self.BLOCK_N_Q = TuningParameter(
            "BLOCK_N_Q", IntegerBounds(min=16, max=128, step=16)
        )
        self.BLOCK_D_KV = TuningParameter(
            "BLOCK_D_KV", IntegerBounds(min=16, max=128, step=16)
        )
        self.BLOCK_N_KV = TuningParameter(
            "BLOCK_N_KV", IntegerBounds(min=16, max=64, step=16)
        )

    @override
    def load_wave_kernel(self):
        config = self.config

        if "f8" in config.dtype:
            base_attention, hyperparams, dynamic_symbols = (
                get_brevitas_pertensor_fp8_attention_kernel(
                    shape=bmnk1k2_to_attention_attributes(config),
                    f8_dtype=dtype_to_torch(config.dtype, self.target),
                    mfma_variant=self.mfma_variant,
                    dynamic_dims=False,
                )
            )
        else:
            base_attention, hyperparams, dynamic_symbols = get_vanilla_attention_kernel(
                shape=bmnk1k2_to_attention_attributes(config),
                mfma_variant=self.mfma_variant,
                dynamic_dims=False,
            )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveKernel(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    @override
    def get_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            use_buffer_ops=True,
        )


class WaveAttentionGQABenchmark(WaveKernelBenchmark):
    config: AttentionConfigBSHD

    def __post_init__(self):
        super().__post_init__()
        self.mfma_variant = TuningParameter(
            "MFMA_VARIANT",
            CategoricalBounds(
                [
                    (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                    (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
                ]
            ),
            initial_value=0,
            include_hyperparam=False,
        )
        self.BLOCK_B = TuningParameter("BLOCK_B", IntegerBounds(min=1, max=1, step=1))
        self.BLOCK_M = TuningParameter(
            "BLOCK_M", IntegerBounds(min=32, max=256, step=8)
        )
        self.BLOCK_N = TuningParameter(
            "BLOCK_N", IntegerBounds(min=16, max=128, step=4)
        )
        self.BLOCK_K2 = TuningParameter(
            "BLOCK_K2", IntegerBounds(min=32, max=256, step=8)
        )

    @override
    def load_wave_kernel(self):
        config = self.config

        base_attention, hyperparams, dynamic_symbols = get_gqa_bshd_attention_kernel(
            shape=bshd_to_attention_attributes(config),
            mfma_variant=self.mfma_variant,
            input_dtype=dtype_to_torch(config.dtype),
            output_dtype=dtype_to_torch("f32"),
        )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveKernel(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    def get_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            iree_launch_async=False,
        )
