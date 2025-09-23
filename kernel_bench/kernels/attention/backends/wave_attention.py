from dataclasses import replace
import torch
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel
from kernel_bench.tuning.hyperparam import (
    TuningParameter,
    CategoricalBounds,
    IntegerBounds,
)
from kernel_bench.core.template import WaveTemplate, WaveKernelBenchmark
from kernel_bench.utils.device_utils import dtype_to_bytes, dtype_to_torch
from wave_lang.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from ..attention_config import (
    AttentionAttributes,
    AttentionConfigBMNK,
    AttentionConfigBSHD,
    AttentionConfigExtend,
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

    def setup_parameters(self):
        config = self.config

        if "f8" in config.dtype:
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

        self.mfma_variant = self.add_param(
            "mfma_variant", mfma_bounds, initial_value=0, include_hyperparam=False
        )
        self.BLOCK_B = self.add_param("BLOCK_B", IntegerBounds(min=1, max=config.B))
        self.BLOCK_M = self.add_param(
            "BLOCK_M", IntegerBounds(min=32, max=config.M, step=4)
        )
        self.BLOCK_N = self.add_param("BLOCK_N", IntegerBounds(min=32, max=config.N))
        self.BLOCK_K2 = self.add_param("BLOCK_K2", IntegerBounds(min=32, max=config.K2))

        bytes_per_el = dtype_to_bytes(config.dtype)
        memory_constraint = (
            self.BLOCK_B * self.BLOCK_K2 * (self.BLOCK_N + 4) * bytes_per_el
            + self.BLOCK_B * self.BLOCK_K2 * (config.K1 + 4) * bytes_per_el
        ) - 65536
        self.add_constraint(memory_constraint, "memory_limit")

    @override
    def load_wave_kernel(self):
        config = self.config

        if "f8" in config.dtype:
            base_attention, hyperparams, dynamic_symbols = (
                get_brevitas_pertensor_fp8_attention_kernel(
                    shape=config.attributes,
                    f8_dtype=dtype_to_torch(config.dtype, self.target),
                    mfma_variant=self.mfma_variant.value,
                    dynamic_dims=False,
                )
            )
        else:
            base_attention, hyperparams, dynamic_symbols = get_vanilla_attention_kernel(
                shape=config.attributes,
                mfma_variant=self.mfma_variant.value,
                dynamic_dims=False,
            )

        hyperparams.update(self._tuning_spec.hyperparams())
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
            use_buffer_ops=True,
        )


class WaveAttentionGQABenchmark(WaveKernelBenchmark):
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

        base_attention, hyperparams, dynamic_symbols = get_gqa_bshd_attention_kernel(
            shape=config.attributes,
            mfma_variant=self.mfma_variant.value,
            input_dtype=dtype_to_torch(config.dtype),
            output_dtype=dtype_to_torch("f32"),
        )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveTemplate(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    def extra_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            iree_launch_async=False,
        )


class WaveExtendAttentionBenchmark(WaveKernelBenchmark):
    config: AttentionConfigExtend

    @override
    def setup_parameters(self):
        mfma_bounds = CategoricalBounds(
            [
                (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
            ]
        )

        self.mfma_variant = self.add_param(
            "mfma_variant", mfma_bounds, initial_value=0, include_hyperparam=False
        )
        # self.add_param("BLOCK_H", IntegerBounds(min=1, max=config.B))
        # self.add_param("BLOCK_N_Q", IntegerBounds(min=32, max=config.M))
        # self.add_param("BLOCK_D_KV", IntegerBounds(min=32, max=config.N))
        # self.add_param("BLOCK_N_KV", IntegerBounds(min=32, max=config.K2))
        # self.add_param("BLOCK_S", IntegerBounds(min=32, max=config.K2))

        # bytes_per_el = dtype_to_bytes(config.dtype)
        # memory_constraint = (
        #     self.BLOCK_B * self.BLOCK_N * (self.BLOCK_K2 + 4) * bytes_per_el
        #     + self.BLOCK_B * self.BLOCK_K2 * (64 + 4) * bytes_per_el
        # ) - 65536
        # self.add_constraint(memory_constraint, "memory_limit")

    @override
    def load_wave_kernel(self):
        config = self.config
        inputs = config.get_inputs()

        base_extend, hyperparams, dynamic_symbols = get_extend_attention_kernel(
            config.attributes,
            self.mfma_variant.value,
            inputs.q_extend.shape,
            inputs.k_extend.shape,
            inputs.v_extend.shape,
            inputs.k_buffer.shape,
            inputs.v_buffer.shape,
            inputs.output.shape,
            input_dtype=dtype_to_torch(config.dtype),
            logit_cap=config.inputs.logit_cap,
        )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveTemplate(
            launchable=base_extend,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    @override
    def extra_compile_options(self):
        return WaveCompileOptions(
            canonicalize=True,
            schedule=SchedulingType.NONE,
            use_buffer_ops=True,
        )
