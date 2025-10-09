from typing import override
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType

from kernel_bench.tuning.hyperparam import CategoricalBounds
from kernel_bench.core.template import WaveTemplate, WaveKernelBenchmark
from kernel_bench.utils.device_utils import dtype_to_torch
from kernel_bench.config.types.attention import AttentionConfigExtend


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
