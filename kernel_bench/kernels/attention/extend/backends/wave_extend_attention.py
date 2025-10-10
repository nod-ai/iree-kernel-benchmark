from typing import override
from kernel_bench.utils.iree_utils import shape_to_iree
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType

from kernel_bench.tuning.hyperparam import CategoricalBounds
from kernel_bench.core.template import WaveTemplate, WaveKernelBenchmark
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

    @override
    def load_wave_kernel(self):
        config = self.config
        inputs = config.get_inputs(self.device_ctx)

        base_extend, hyperparams, dynamic_symbols = get_extend_attention_kernel(
            config.attributes,
            self.mfma_variant.value,
            inputs.q_extend.shape,
            inputs.k_extend.shape,
            inputs.v_extend.shape,
            inputs.k_buffer.shape,
            inputs.v_buffer.shape,
            inputs.output.shape,
            input_dtype=self.device_ctx.dtype_to_torch(config.dtype),
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

    @override
    def get_runtime_args(self):
        if not self.config.inputs:
            self.config.get_inputs(self.device_ctx)

        inputs = self.config.inputs
        in_dtype = self.config.dtype

        bench_inputs = [
            shape_to_iree(inputs.q_extend.shape, in_dtype, self.device_ctx),
            shape_to_iree(inputs.k_extend.shape, in_dtype, self.device_ctx),
            shape_to_iree(inputs.v_extend.shape, in_dtype, self.device_ctx),
            shape_to_iree(inputs.k_buffer.shape, in_dtype, self.device_ctx),
            shape_to_iree(inputs.v_buffer.shape, in_dtype, self.device_ctx),
            shape_to_iree(inputs.qo_indptr.shape, "i32", self.device_ctx),
            shape_to_iree(inputs.kv_indptr.shape, "i32", self.device_ctx),
            shape_to_iree(inputs.kv_indices.shape, "i32", self.device_ctx),
            shape_to_iree(inputs.output.shape, "f32", self.device_ctx),
            shape_to_iree(inputs.max_len_extend, "i32", self.device_ctx),
        ]

        runtime_args = [f"--input={shape}" for shape in bench_inputs]
        runtime_args += ["--function=isolated_benchmark"]
        return runtime_args
