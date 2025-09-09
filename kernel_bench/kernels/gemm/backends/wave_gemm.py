from typing import override
import torch
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul

from kernel_bench.tuning.hyperparam import (
    TuningParameter,
    CategoricalBounds,
    IntegerBounds,
)
from kernel_bench.core.template import WaveKernelBenchmark, WaveKernel
from kernel_bench.utils.device_utils import dtype_to_bits, dtype_to_torch
from ..gemm_utils import GemmConfig


class WaveGemmBenchmark(WaveKernelBenchmark):
    config: GemmConfig

    def __post_init__(self):
        super().__post_init__()

        dtype = dtype_to_torch(self.config.operand_element_type)
        bitwidth = dtype_to_bits(self.config.operand_element_type)

        if bitwidth == 8:
            mfma_options = [(MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K8_F16)]
        elif dtype == torch.float16:
            mfma_options = [
                (MMAType.F32_32x32x16_F16, MMAType.F32_32x32x16_F16),
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_16x16x32_F16, MMAType.F32_16x16x32_F16),
                (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
                (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x16_K8_F16),
            ]
        elif dtype == torch.bfloat16:
            mfma_options = [
                (MMAType.F32_32x32x16_BF16, MMAType.F32_32x32x16_BF16),
                (MMAType.F32_16x16x32_BF16, MMAType.F32_16x16x32_BF16),
            ]

        self.mfma_variant = TuningParameter(
            "MFMA_VARIANT",
            CategoricalBounds(mfma_options),
            initial_value=0,
            include_hyperparam=False,
        )
        self.BLOCK_M = TuningParameter(
            "BLOCK_M", IntegerBounds(min=16, max=256, step=8), initial_value=128
        )
        self.BLOCK_N = TuningParameter(
            "BLOCK_N", IntegerBounds(min=16, max=256, step=8), initial_value=256
        )
        self.BLOCK_K = TuningParameter(
            "BLOCK_K", IntegerBounds(min=16, max=128, step=4), initial_value=64
        )
        self.GROUP_SIZE_M = TuningParameter(
            "GROUP_SIZE_M", IntegerBounds(min=8, max=16, step=8), initial_value=8
        )

    @override
    def load_wave_kernel(self):
        config = self.config

        input_dtype = dtype_to_torch(config.operand_element_type)
        output_dtype = dtype_to_torch(config.result_element_type)
        quant_dtype = None
        if "f8" in config.operand_element_type:
            quant_dtype = input_dtype
            input_dtype = dtype_to_torch("f16")

        tA, tB = config.tA, config.tB

        base_gemm, hyperparams = get_reordered_matmul(
            config.M,
            config.N,
            config.K,
            self.BLOCK_M,
            self.BLOCK_N,
            self.BLOCK_K,
            self.GROUP_SIZE_M,
            mfma_variant=self.mfma_variant,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            quantized_dtype=quant_dtype,
            tA=tA,
            tB=tB,
        )

        hyperparams.update(get_default_scheduling_params())

        return WaveKernel(launchable=base_gemm, hyperparams=hyperparams)

    @override
    def get_compile_options(self):
        tA = self.config.tA
        tB = self.config.tB

        use_scheduling = SchedulingType.PREFETCH

        return WaveCompileOptions(
            canonicalize=True,
            schedule=use_scheduling,
            use_buffer_ops=True,
            multi_buffer_count=(
                2
                if use_scheduling in [SchedulingType.FOUR_STAGE, SchedulingType.MODULO]
                else None
            ),
        )
