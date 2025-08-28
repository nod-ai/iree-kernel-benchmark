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


def get_block_size(default: int, upper_bound: int):
    block_size = min(default, upper_bound)
    exp = int(np.log2(block_size))
    block_size_pow2 = pow(2, exp)
    return max(1, block_size_pow2)


class WaveGemmBenchmark(KernelBenchmark):
    @override
    def compile_kernel(
        self,
        config: GemmConfig,
        mlir_path,
        vmfb_path,
        extra_compiler_args=...,
        spec=None,
        mfma_variant=None,
    ):
        try:
            use_quantized = "f8" in config.operand_element_type

            if not mfma_variant:
                if use_quantized:
                    mfma_variant = (
                        MMAType.F32_32x32x16_K8_F16,
                        MMAType.F32_32x32x16_F8,
                    )
                else:
                    mfma_variant = (
                        MMAType.F32_32x32x16_K8_F16,
                        MMAType.F32_32x32x16_K8_F16,
                    )

            BLOCK_M = get_block_size(128, config.M)
            BLOCK_N = get_block_size(256, config.N)
            BLOCK_K = get_block_size(64, config.K)
            GROUP_SIZE_M = 8

            # print(BLOCK_M, BLOCK_N, BLOCK_K)

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
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                mfma_variant=mfma_variant,
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                quant_dtype=quant_dtype,
                tA=tA,
                tB=tB,
            )

            use_scheduling = (
                SchedulingType.PREFETCH if tA + tB == "NT" else SchedulingType.NONE
            )

            if spec:
                spec = GemmTuningSpec(spec)
                hyperparams.update(spec.hyperparams())
            hyperparams.update(get_default_scheduling_params())

            compile_options = WaveCompileOptions(
                subs=hyperparams,
                canonicalize=True,
                create_vmfb_file=vmfb_path,
                run_bench=False,
                target=self.target,
                schedule=use_scheduling,
                use_buffer_ops=True,
                iree_launch_async=False,
                multi_buffer_count=(
                    2
                    if use_scheduling
                    in [SchedulingType.FOUR_STAGE, SchedulingType.MODULO]
                    else None
                ),
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
