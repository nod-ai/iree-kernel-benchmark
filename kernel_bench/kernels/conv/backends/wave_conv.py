from typing import override
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.templates.conv import get_igemm_conv2d
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType

from kernel_bench.core.template import WaveKernelBenchmark, WaveTemplate
from kernel_bench.tuning.hyperparam import IntegerBounds
from kernel_bench.utils import *
from ..conv_utils import (
    ConvConfig,
    get_iree_conv_img_shape,
    get_iree_conv_kernel_shape,
    get_iree_conv_out_shape,
)


class WaveConvBenchmark(WaveKernelBenchmark):
    config: ConvConfig

    def setup_parameters(self):
        self.BLOCK_M = self.add_param("BLOCK_M", IntegerBounds(min=16, max=256, step=8))
        self.BLOCK_N = self.add_param("BLOCK_N", IntegerBounds(min=16, max=256, step=8))
        self.BLOCK_K = self.add_param("BLOCK_K", IntegerBounds(min=16, max=128, step=4))
        self.ELEMS_PER_THREAD = self.add_param(
            "ELEMS_PER_THREAD", IntegerBounds(min=4, max=4, step=1)
        )

        # Add some example constraints for conv operations
        # Ensure reasonable block sizes for convolution workloads
        memory_constraint = self.BLOCK_M * self.BLOCK_N <= 8192
        self.add_constraint(memory_constraint, "conv_memory_limit")

    @override
    def load_wave_kernel(self):
        config = self.config

        op_type, layout = config.decode_op()

        in_h = config.H * config.S + config.P - 1
        in_w = config.W * config.S + config.Q - 1
        if op_type == "conv_2d":
            conv, hyperparams = get_igemm_conv2d(
                layout=layout,
                n=config.N,
                h=in_h,
                w=in_w,
                c=config.C,
                hf=config.P,
                wf=config.Q,
                nf=config.F,
                stride=config.S,
                input_dtype=self._convert_dtype(config.input_dtype),
                output_dtype=self._convert_dtype(config.output_dtype),
            )
        else:
            raise Exception(f"Op type {op_type} is not supported for wave convolutions")

        hyperparams.update(self._tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveTemplate(launchable=conv, hyperparams=hyperparams)

    @override
    def extra_compile_options(self):
        return WaveCompileOptions(
            canonicalize=True,
            schedule=SchedulingType.NONE,
        )

    def get_runtime_args(self):
        image_shape = get_iree_conv_img_shape(self.config, self.device_ctx)
        filter_shape = get_iree_conv_kernel_shape(self.config, self.device_ctx)
        out_shape = get_iree_conv_out_shape(self.config, self.device_ctx)
        runtime_args = [
            f"--input={image_shape}",
            f"--input={filter_shape}",
            f"--input={out_shape}",
            "--function=isolated_benchmark",
        ]
        return runtime_args

    def _convert_dtype(self, dtype: str):
        dtypes = {
            "i8": tkl.i8,
            "i16": tkl.i16,
            "i32": tkl.i32,
            "i64": tkl.i64,
            "f16": tkl.f16,
            "f32": tkl.f32,
            "f64": tkl.f64,
            "bf16": tkl.bf16,
        }
        return dtypes[dtype]
