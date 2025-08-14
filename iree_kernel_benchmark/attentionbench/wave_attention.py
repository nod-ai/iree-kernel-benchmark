from ..utils import *
from .attention_config import (
    AttentionAttributes,
    AttentionConfigBMNK,
    AttentionConfigBSHD,
    bmnk1k2_to_attention_attributes,
    bshd_to_attention_attributes,
)
from .attention_utils import (
    AttentionBMNKTuningSpec,
    AttentionBSHDTuningSpec,
)
from pathlib import Path
from typing import Optional, override

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
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


class WaveAttentionMHABenchmark(KernelBenchmark):
    @override
    def compile_kernel(
        self,
        config: AttentionConfigBMNK,
        mlir_path,
        vmfb_path,
        extra_compiler_args=...,
        mfma_variant=None,
        spec=None,
    ):
        try:
            if not mfma_variant:
                mfma_variant = (
                    MMAType.F32_32x32x16_K8_F16,
                    MMAType.F32_32x32x8_F16,
                )

            base_attention, hyperparams, dynamic_symbols = get_vanilla_attention_kernel(
                shape=bmnk1k2_to_attention_attributes(config),
                mfma_variant=mfma_variant,
                dynamic_dims=False,
            )

            if spec:
                hyperparams.update(spec.hyperparams())

            hyperparams.update(get_default_scheduling_params())

            compile_options = WaveCompileOptions(
                subs=hyperparams,
                schedule=SchedulingType.NONE,
                canonicalize=True,
                create_vmfb_file=vmfb_path,
                iree_launch_async=False,
                backend="rocm",
                target=self.target,
                use_buffer_load_ops=True,
                use_buffer_store_ops=True,
                use_stride_cache_swizzle=True,
                print_ir_after_all=self.dump_dir is not None,
            )

            if self.dump_dir:
                dump_file = self.dump_dir / "wave" / (config.get_name() + ".debug.mlir")
                with redirect_stderr_to_file(dump_file):
                    result = wave_compile(compile_options, base_attention)
            else:
                result = wave_compile(compile_options, base_attention)

            with open(mlir_path, "w") as mlir_out:
                mlir_out.write(result.asm)

            return True

        except Exception as e:
            print(f"Failed to compile {config.get_name()}: {e}")
            return False


class WaveAttentionGQABenchmark(KernelBenchmark):
    @override
    def compile_kernel(
        self,
        config: AttentionConfigBSHD,
        mlir_path,
        vmfb_path,
        extra_compiler_args=...,
        mfma_variant=None,
        spec=None,
    ):
        try:
            if not mfma_variant:
                mfma_variant = (
                    MMAType.F32_32x32x16_K8_F16,
                    MMAType.F32_32x32x8_F16,
                )

            base_attention, hyperparams, dynamic_symbols = (
                get_gqa_bshd_attention_kernel(
                    shape=bshd_to_attention_attributes(config),
                    mfma_variant=mfma_variant,
                    input_dtype=dtype_to_torch(config.dtype),
                    output_dtype=dtype_to_torch("f32"),
                )
            )

            if spec:
                hyperparams.update(spec.hyperparams())
            hyperparams.update(get_default_scheduling_params())

            compile_options = WaveCompileOptions(
                subs=hyperparams,
                schedule=SchedulingType.NONE,
                dynamic_symbols=dynamic_symbols,
                canonicalize=True,
                create_vmfb_file=vmfb_path,
                iree_launch_async=False,
                backend="rocm",
                target=self.target,
                print_ir_after_all=self.dump_dir is not None,
            )

            if self.dump_dir:
                dump_file = self.dump_dir / "wave" / (config.get_name() + ".debug.mlir")
                with redirect_stderr_to_file(dump_file):
                    result = wave_compile(compile_options, base_attention)
            else:
                result = wave_compile(compile_options, base_attention)

            with open(mlir_path, "w") as mlir_out:
                mlir_out.write(result.asm)

            return True

        except Exception as e:
            print(f"Failed to compile {config.get_name()}: {e}")
            return False
