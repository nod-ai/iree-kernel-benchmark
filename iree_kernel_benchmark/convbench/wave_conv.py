from dataclasses import asdict
from ..utils import *
from pathlib import Path
from typing import Optional, override
from .conv_utils import ConvConfig
import traceback

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.templates.conv import get_igemm_conv2d
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


class WaveConvBenchmark(KernelBenchmark):
    @override
    def compile_kernel(
        self,
        config: ConvConfig,
        mlir_path,
        vmfb_path,
        extra_compiler_args=...,
        mfma_variant=None,
        spec=None,
    ):
        try:
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
                return False

            if spec:
                hyperparams.update(spec.hyperparams())

            options = WaveCompileOptions(
                subs=hyperparams,
                canonicalize=True,
                create_vmfb_file=vmfb_path,
                schedule=SchedulingType.NONE,
                iree_launch_async=False,
                backend="rocm",
                target=self.target,
            )
            result = wave_compile(options, conv)
            with open(mlir_path, "w") as f:
                f.write(result.asm)

            return True

        except Exception as e:
            print(f"Failed to compile {config.get_name()}: {e}")
            return False

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
