import torch
from kernel_bench.config.types.attention.vanilla_attention_config import (
    bmnk1k2_to_attention_attributes,
)
from kernel_bench.kernels.attention.vanilla.attention_utils import (
    get_iree_attention_shapes,
)
from kernel_bench.tuning.hyperparam import CategoricalBounds, IntegerBounds
from kernel_bench.core.template import WaveTemplate, WaveKernelBenchmark
from wave_lang.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from kernel_bench.config.types.attention import AttentionConfigBMNK
from typing import override

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


class WaveVanillaAttentionBenchmark(WaveKernelBenchmark):
    config: AttentionConfigBMNK

    def setup_parameters(self):
        config = self.config

        if config.dtype == "f8":
            mfma_bounds = CategoricalBounds(
                [
                    (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
                    (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
                ]
            )
        else:
            mfma_bounds = CategoricalBounds(
                [
                    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                    (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
                ]
            )

        mma_val = 0
        BLOCK_B_val = 1
        BLOCK_M_val = 128
        BLOCK_N_val = min(128, config.N)
        BLOCK_K2_val = 64

        self.mfma_variant = self.add_param(
            "mfma_variant", mfma_bounds, initial_value=mma_val, include_hyperparam=False
        )
        self.BLOCK_B = self.add_param(
            "BLOCK_B",
            IntegerBounds(min=1, max=config.B),
            initial_value=BLOCK_B_val,
            clamp_value=True,
        )
        self.BLOCK_M = self.add_param(
            "BLOCK_M",
            IntegerBounds(min=32, max=config.M, step=4),
            initial_value=BLOCK_M_val,
            clamp_value=True,
        )
        self.BLOCK_N = self.add_param(
            "BLOCK_N",
            IntegerBounds(min=32, max=config.N),
            initial_value=BLOCK_N_val,
            clamp_value=True,
        )
        self.BLOCK_K2 = self.add_param(
            "BLOCK_K2",
            IntegerBounds(min=32, max=config.K2),
            initial_value=BLOCK_K2_val,
            clamp_value=True,
        )

        bytes_per_el = self.device_ctx.resolve_dtype(config.dtype).num_bytes()
        memory_constraint = (
            self.BLOCK_B * self.BLOCK_K2 * (self.BLOCK_N + 4) * bytes_per_el
            + self.BLOCK_B * self.BLOCK_K2 * (config.K1 + 4) * bytes_per_el
        ) - 65536
        self.add_constraint(memory_constraint, "memory_limit")

    @override
    def load_wave_kernel(self):
        config = self.config
        shape = bmnk1k2_to_attention_attributes(config)

        if "f8" in config.dtype:
            base_attention, hyperparams, dynamic_symbols = (
                get_brevitas_pertensor_fp8_attention_kernel(
                    shape=shape,
                    f8_dtype=self.device_ctx.dtype_to_torch(config.dtype),
                    mfma_variant=self.mfma_variant.value,
                    dynamic_dims=False,
                )
            )
        else:
            base_attention, hyperparams, dynamic_symbols = get_vanilla_attention_kernel(
                shape=shape,
                mfma_variant=self.mfma_variant.value,
                dynamic_dims=False,
            )

        hyperparams.update(self._tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        template = WaveTemplate(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

        return template

    @override
    def extra_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            use_buffer_ops=True,
        )

    @override
    def get_runtime_args(self):
        config = self.config
        in_dtype = "f16" if config.dtype == "f8" else config.dtype
        out_dtype = "f32"
        query_shape, key_shape, value_shape, output_shape = get_iree_attention_shapes(
            config, self.device_ctx, in_dtype, out_dtype
        )

        runtime_args = [
            f"--input={shape}"
            for shape in [query_shape, key_shape, value_shape, output_shape]
        ]
        runtime_args += ["--function=isolated_benchmark"]
        return runtime_args
