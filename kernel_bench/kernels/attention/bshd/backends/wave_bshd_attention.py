from kernel_bench.config.types.attention.bshd_attention_config import (
    bshd_to_attention_attributes,
)
from kernel_bench.tuning.hyperparam import CategoricalBounds, IntegerBounds
from kernel_bench.core.template import WaveTemplate, WaveKernelBenchmark
from kernel_bench.config.types.attention import AttentionConfigBSHD
from typing import override

from kernel_bench.utils.iree_utils import shape_to_iree
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.templates.gqa_vanilla_attention import (
    get_gqa_bshd_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


class WaveBSHDAttentionBenchmark(WaveKernelBenchmark):
    config: AttentionConfigBSHD

    def setup_parameters(self):
        self.mfma_variant = self.add_param(
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
        self.BLOCK_B = self.add_param("BLOCK_B", IntegerBounds(min=1, max=1, step=1))
        self.BLOCK_H = self.add_param("BLOCK_H", IntegerBounds(min=1, max=2, step=1))
        self.BLOCK_N_Q = self.add_param(
            "BLOCK_N_Q", IntegerBounds(min=16, max=128, step=16)
        )
        self.BLOCK_D_KV = self.add_param(
            "BLOCK_D_KV", IntegerBounds(min=16, max=128, step=16)
        )
        self.BLOCK_N_KV = self.add_param(
            "BLOCK_N_KV", IntegerBounds(min=16, max=64, step=16)
        )

    @override
    def load_wave_kernel(self):
        config = self.config
        shape = bshd_to_attention_attributes(config)

        base_attention, hyperparams, dynamic_symbols = get_gqa_bshd_attention_kernel(
            shape=shape,
            mfma_variant=self.mfma_variant.value,
            input_dtype=self.device_ctx.dtype_to_torch(config.dtype),
            output_dtype=self.device_ctx.dtype_to_torch("f32"),
        )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveTemplate(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    @override
    def extra_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            iree_launch_async=False,
        )

    @override
    def get_runtime_args(self):
        config = self.config
        in_dtype = "f16" if config.dtype == "f8" else config.dtype
        out_dtype = "f32"

        query_shape = shape_to_iree(
            (config.B, config.N_Q, config.H, config.D_Q), in_dtype, self.device_ctx
        )
        key_shape = shape_to_iree(
            (config.B, config.N_KV, config.H_KV, config.D_Q),
            in_dtype,
            self.device_ctx,
        )
        value_shape = shape_to_iree(
            (config.B, config.N_KV, config.H_KV, config.D_KV),
            in_dtype,
            self.device_ctx,
        )
        output_shape = shape_to_iree(
            (config.B, config.N_Q, config.H, config.D_KV), out_dtype, self.device_ctx
        )

        runtime_args = [
            f"--input={shape}"
            for shape in [query_shape, key_shape, value_shape, output_shape]
        ]
        runtime_args += ["--function=isolated_benchmark"]
        return runtime_args
