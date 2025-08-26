from dataclasses import asdict
import torch
import sympy
from typing import Any, Dict, List, Optional, Tuple, Type, override

# from wave_lang.kernel.wave.wave import LaunchableWave
# from wave_lang.kernel.lang import IndexSymbol
# from wave_lang.kernel.wave.constraints import MMAType
# from wave_lang.kernel.wave.compile import wave_compile
# from wave_lang.kernel.wave.compile_options import WaveCompileOptions

# import iree.runtime as ireert

# from .tuning import TuningConstraint, tune_kernels_parallel
# from .bench_utils import redirect_stderr_to_file, OpConfig


DTYPE_TO_TORCH = {
    "bf16": torch.bfloat16,
    "f8e5m2": torch.float8_e5m2,
    "f8e5m2fnuz": torch.float8_e5m2fnuz,
    "f8e4m3fn": torch.float8_e4m3fn,
    "f8e4m3fnuz": torch.float8_e4m3fnuz,
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}

DTYPE_TO_BITS = {
    "bf16": 16,
    "f8e5m2": 8,
    "f8e5m2fnuz": 8,
    "f8e4m3fn": 8,
    "f8e4m3fnuz": 8,
    "f16": 16,
    "f32": 32,
    "f64": 64,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "bool": 1,
}


def dtype_to_torch(dtype: str):
    dtype = dtype.lower().strip().replace("_", "")
    return DTYPE_TO_TORCH[dtype]


def dtype_to_bits(dtype: str):
    dtype = dtype.lower().strip().replace("_", "")
    return DTYPE_TO_BITS[dtype]


def dtype_to_bytes(dtype: str):
    return max(1, dtype_to_bits(dtype) // 8)


# class WaveKernelBenchmark(KernelBenchmark):
#     backend = "wave"

#     def load_kernel(
#         self,
#         config: OpConfig,
#         spec: TuningSpec = None,
#         mfma_variant: Tuple[MMAType] = None,
#     ) -> Tuple[LaunchableWave, Dict[IndexSymbol, Any]]:
#         pass

#     def get_extra_compile_options(self) -> WaveCompileOptions | Dict[str, Any]:
#         return {}

#     @override
#     def compile_kernel(
#         self,
#         config: OpConfig,
#         mlir_path,
#         vmfb_path,
#         extra_compiler_args=...,
#         mfma_variant=None,
#         spec=None,
#     ):
#         try:
#             wave_kernel, hyperparams = self.load_kernel(config, spec, mfma_variant)

#             extra_compile_options = self.get_extra_compile_options()
#             extra_compile_options = (
#                 asdict(extra_compile_options)
#                 if isinstance(extra_compile_options, WaveCompileOptions)
#                 else extra_compile_options
#             )
#             compile_options = WaveCompileOptions(
#                 subs=hyperparams,
#                 canonicalize=True,
#                 create_vmfb_file=vmfb_path,
#                 run_bench=False,
#                 iree_launch_async=False,
#                 target=self.target,
#                 **extra_compile_options,
#             )

#             if self.dump_dir:
#                 dump_file = self.dump_dir / "wave" / (config.get_name() + ".debug.mlir")
#                 with redirect_stderr_to_file(dump_file):
#                     result = wave_compile(compile_options, wave_kernel)
#             else:
#                 result = wave_compile(compile_options, wave_kernel)

#             with open(mlir_path, "w") as mlir_out:
#                 mlir_out.write(result.asm)

#             return True

#         except Exception as e:
#             print(f"Failed to compile {config.get_name()}: {e}")
#             return False
