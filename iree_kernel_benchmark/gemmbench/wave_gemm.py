from ..utils import *
from .gemm_utils import GemmConfig, GemmTuningSpec
from pathlib import Path
from typing import Optional, override

from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul


class WaveGemmBenchmark(KernelBenchmark):
    @override
    def compile_kernel(
        self,
        config: GemmConfig,
        mlir_path,
        vmfb_path,
        extra_compiler_args=...,
        mfma_variant=None,
        spec=None,
    ):
        try:
            if not mfma_variant:
                mfma_variant = MMAType.F32_32x32x8_F16

            BLOCK_M = 128
            BLOCK_N = 256
            BLOCK_K = 64
            GROUP_SIZE_M = 8

            base_gemm, hyperparams = get_reordered_matmul(
                config.M,
                config.N,
                config.K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                mfma_variant=mfma_variant,
            )

            if spec:
                hyperparams.update(spec.hyperparams())
            hyperparams.update(get_default_scheduling_params())

            compile_options = WaveCompileOptions(
                subs=hyperparams,
                canonicalize=True,
                create_vmfb_file=vmfb_path,
                run_bench=False,
                target=self.target,
                schedule=SchedulingType.PREFETCH,
                use_buffer_load_ops=True,
                use_buffer_store_ops=True,
                use_stride_cache_swizzle=True,
                iree_launch_async=False,
            )

            if self.dump_dir:
                dump_file = self.dump_dir / "wave" / (config.get_name() + ".debug.mlir")
                with redirect_stderr_to_file(dump_file):
                    result = wave_compile(compile_options, base_gemm)
            else:
                result = wave_compile(compile_options, base_gemm)

            with open(mlir_path, "w") as mlir_out:
                mlir_out.write(result.asm)

            return True

        except Exception as e:
            print(f"Failed to compile {config.get_name()}: {e}")
            return False
