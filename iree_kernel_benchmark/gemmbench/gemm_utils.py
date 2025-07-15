from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    from iree.turbine.kernel.wave.constraints import MMAType
    import iree.turbine.kernel as tk
    import iree.turbine.kernel.lang as tkl
    import iree.turbine.kernel.wave as tkw
    from iree.turbine.kernel.lang.global_symbols import *
    from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
    from iree.turbine.kernel.wave.utils.general_utils import (
        get_default_scheduling_params,
    )
    from iree.turbine.kernel.wave.scheduling.schedule_enums import SchedulingType
except ImportError as e:
    TURBINE_AVAILABLE = False
    turbine_import_error = e
else:
    TURBINE_AVAILABLE = True

from ..utils import *
import os
import traceback
from iree.compiler import ir
from iree.compiler.dialects import arith, func, linalg, tensor

kDynamic = ir.ShapedType.get_dynamic_size()


def num_bytes(dtype: str) -> int:
    dtype_to_bytes = {
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "f8E4M3FNUZ": 1,
        "f8E5M2FNUZ": 1,
        "f8E4M3FN": 1,
        "f8E5M2": 1,
        "i8": 1,
        "i32": 4,
    }
    return dtype_to_bytes[dtype]


@dataclass
class GemmConfig:
    # Note that M, N and K may be set to kDynamic, a special value
    M: int
    N: int
    K: int
    tA: str
    tB: str
    operand_element_type: str
    accumulator_element_type: str
    result_element_type: str
    # runtime_dim subtitutes for any dynamic dims when executing.
    # TODO: It would be better if we could execute the same compiled dynamic
    #       kernel for a series of different sizes, rather than duplicating the
    #       GemmConfig. The current design's advantage is that no changes have
    #       to be made to the execution logic (looks just like a static shape).
    runtime_dim: Optional[int] = None

    def get_name(self) -> str:
        M = self.M if self.M != kDynamic else "D"
        N = self.N if self.N != kDynamic else "D"
        K = self.K if self.K != kDynamic else "D"
        name = f"gemm_{M}_{N}_{K}_{self.operand_element_type}_{self.accumulator_element_type}"
        if self.tA == "T":
            name += "_tA"
        elif self.tB == "T":
            name += "_tB"
        if self.runtime_dim is not None:
            name += f"_D={self.runtime_dim}"
        return name

    def get_runtime_dims(self) -> Tuple[int, int, int]:
        """
        Get concrete dims to use when executing this kernel.
        """
        M = self.M if self.M != kDynamic else self.runtime_dim
        N = self.N if self.N != kDynamic else self.runtime_dim
        K = self.K if self.K != kDynamic else self.runtime_dim
        return M, N, K

    def get_inp1(self) -> str:
        M, N, K = self.get_runtime_dims()
        if self.tA == "T":
            return f"{K}x{M}x{self.operand_element_type}"
        return f"{M}x{K}x{self.operand_element_type}"

    def get_inp2(self) -> str:
        M, N, K = self.get_runtime_dims()
        if self.tB == "T":
            return f"{N}x{K}x{self.operand_element_type}"
        return f"{K}x{N}x{self.operand_element_type}"

    def get_out(self) -> str:
        M, N, K = self.get_runtime_dims()
        return f"{M}x{N}x{self.result_element_type}"

    def get_byte_count(self) -> int:
        operand_bytes_per_element = num_bytes(self.operand_element_type)
        result_bytes_per_element = num_bytes(self.result_element_type)
        M, N, K = self.get_runtime_dims()
        byte_count_input = (M + N) * K * operand_bytes_per_element
        byte_count_output = (M * N) * result_bytes_per_element
        return byte_count_input + byte_count_output

    def get_flops(self) -> int:
        M, N, K = self.get_runtime_dims()
        flops = 2 * M * N * K
        return flops


def _convert_dtype_to_mlir(dtype: str) -> ir.Type:
    dtypes = {
        "i8": lambda: ir.IntegerType.get_signless(8),
        "i16": lambda: ir.IntegerType.get_signless(16),
        "i32": lambda: ir.IntegerType.get_signless(32),
        "i64": lambda: ir.IntegerType.get_signless(64),
        "f8E4M3FNUZ": lambda: ir.Float8E4M3FNUZType.get(),
        "f8E5M2FNUZ": lambda: ir.Float8E5M2FNUZType.get(),
        "f8E4M3FN": lambda: ir.Float8E4M3FNType.get(),
        "f8E5M2": lambda: ir.Float8E5M2Type.get(),
        "f16": lambda: ir.F16Type.get(),
        "f32": lambda: ir.F32Type.get(),
        "f64": lambda: ir.F64Type.get(),
        "bf16": lambda: ir.BF16Type.get(),
    }
    return dtypes[dtype]()


def generate_mlir(config: GemmConfig):
    K = config.K
    M = config.M
    N = config.N
    tA = config.tA
    tB = config.tB

    with ir.Location.name(config.get_name()):
        operand_element_type = _convert_dtype_to_mlir(config.operand_element_type)
        acc_element_type = _convert_dtype_to_mlir(config.accumulator_element_type)
        result_element_type = _convert_dtype_to_mlir(config.result_element_type)
        is_integer = isinstance(operand_element_type, ir.IntegerType)
        literal_zero = (
            ir.IntegerAttr.get(acc_element_type, 0)
            if is_integer
            else ir.FloatAttr.get(acc_element_type, 0.0)
        )

        # Transpose A
        if tA == "T":
            arg0_type = ir.RankedTensorType.get([K, M], operand_element_type)
            arg0_M_idx = 1
            arg1_type = ir.RankedTensorType.get([K, N], operand_element_type)
            arg1_N_idx = 1
        # Transpose B
        elif tB == "T":
            arg0_type = ir.RankedTensorType.get([M, K], operand_element_type)
            arg0_M_idx = 0
            arg1_type = ir.RankedTensorType.get([N, K], operand_element_type)
            arg1_N_idx = 0
        # "Normal" path (can't transpose both)
        else:
            assert tA == "N" and tB == "N"
            arg0_type = ir.RankedTensorType.get([M, K], operand_element_type)
            arg0_M_idx = 0
            arg1_type = ir.RankedTensorType.get([K, N], operand_element_type)
            arg1_N_idx = 1
        result_type = ir.RankedTensorType.get([M, N], result_element_type)

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(arg0_type, arg1_type)
            def main(arg0, arg1):
                zero_element = arith.constant(
                    value=literal_zero, result=acc_element_type
                )
                if M == kDynamic:
                    M_dynamic_dim_idx = arith.constant(
                        value=arg0_M_idx, result=ir.IndexType.get()
                    )
                    M_dynamic_dim = tensor.dim(arg0, M_dynamic_dim_idx)
                if N == kDynamic:
                    N_dynamic_dim_idx = arith.constant(
                        value=arg1_N_idx, result=ir.IndexType.get()
                    )
                    N_dynamic_dim = tensor.dim(arg1, N_dynamic_dim_idx)

                empty_tensor = tensor.empty(
                    element_type=acc_element_type,
                    sizes=[
                        M_dynamic_dim if M == kDynamic else M,
                        N_dynamic_dim if N == kDynamic else N,
                    ],
                )
                filled_tensor = linalg.fill(zero_element, outs=[empty_tensor])

                if tA == "T":
                    acc = linalg.matmul_transpose_a(arg0, arg1, outs=[filled_tensor])
                elif tB == "T":
                    acc = linalg.matmul_transpose_b(arg0, arg1, outs=[filled_tensor])
                else:
                    acc = linalg.matmul(arg0, arg1, outs=[filled_tensor])

                if acc_element_type == result_element_type:
                    return acc
                if is_integer:
                    return arith.trunci(result_type, acc)
                return arith.truncf(result_type, acc)

        return f"{module}"


@dataclass
class TkTunedConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    RATIO_M: int
    RATIO_N: int
    WAVES_PER_EU: int
    MMA_UNITS: int
    SHARED_UNITS: int
    GLOBAL_UNITS: int
    DELAY_MMA: int
    DELAY_SHARED: int
    DELAY_GLOBAL: int


def get_tk_tuned_config(config: GemmConfig) -> TkTunedConfig:
    if config.M == 2048 and config.N == 10240 and config.K == 1280:
        return TkTunedConfig(128, 320, 32, 2, 2, 2, 2, 2, 2, 1, 1, 2)
    if config.M == 2048 and config.N == 1280 and config.K == 1280:
        return TkTunedConfig(64, 64, 64, 2, 2, 1, 2, 1, 1, 1, 1, 2)
    if config.M == 2048 and config.N == 1280 and config.K == 5120:
        return TkTunedConfig(128, 80, 128, 4, 1, 1, 4, 2, 2, 1, 1, 2)
    if config.M == 128 and config.N == 1280 and config.K == 2048:
        return TkTunedConfig(64, 64, 128, 2, 2, 1, 8, 2, 2, 1, 1, 2)
    if config.M == 8192 and config.N == 5120 and config.K == 640:
        return TkTunedConfig(128, 128, 32, 2, 2, 1, 4, 2, 2, 1, 1, 2)
    # Default config
    return TkTunedConfig(64, 64, 32, 2, 2, 1, 2, 2, 2, 1, 1, 2)


def _convert_dtype_to_tk(dtype: str):
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


def generate_tk_mlir(config: GemmConfig, vmfb_file: Path):
    # TODO: Enable waves_per_eu
    # TODO: Use scheduling barriers with LLVM patch
    tc = get_tk_tuned_config(config)
    assert config.operand_element_type == "f16", "Unsupported problem"
    assert config.accumulator_element_type == "f32", "Unsupported problem"

    res_dtype = _convert_dtype_to_tk(config.result_element_type)
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / tc.RATIO_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / tc.RATIO_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(tc.RATIO_M, tc.RATIO_N, 1)
        )
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, res_dtype],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        if res_dtype != tkl.f32:
            repeat = tkw.cast(repeat, res_dtype)

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    shape = [config.M, config.N, config.K]
    schedule = SchedulingType.MODULO if config.K < 4096 else SchedulingType.NONE

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_M: tc.BLOCK_M,
        BLOCK_N: tc.BLOCK_N,
        BLOCK_K: tc.BLOCK_K,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    # TODO: Scheduling is taking too long time with large K.
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        backend="rocm",
        target="gfx942",
        schedule=schedule,
    )
    result = wave_compile(options, gemm)
    return result.asm


def compile_gemm_config(
    config: GemmConfig,
    kernel_dir: Path,
    vmfb_dir: Path,
    target,
    extra_compiler_args,
    tk,
) -> tuple[Path, Optional[Path]]:
    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")
    dump_file = kernel_dir / (config.get_name() + ".stderr.mlir")

    if not os.path.exists(vmfb_dir):
        os.makedirs(vmfb_dir)

    # Generate mlir content
    if tk and not TURBINE_AVAILABLE:
        raise ValueError(
            f"Can't compile TK benchmark because of a failed import (most likely iree.turbine is missing): {turbine_import_error}"
        )
    if tk:
        try:
            mlir_content = generate_tk_mlir(config, vmfb_file)
        except Exception as e:
            # traceback.print_exc()
            error_file = vmfb_dir / (config.get_name() + "_error.txt")
            print(
                f"Failed to compile {config.get_name()}. Error dumped in {error_file}"
            )
            with open(error_file, "w") as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            return mlir_file, None
    else:
        with ir.Context():
            mlir_content = generate_mlir(config)

    # Write MLIR content to file
    with open(mlir_file, "w") as f:
        f.write(mlir_content)

    if tk:
        return mlir_file, vmfb_file

    # Compile MLIR to VMFB
    exec_args = [
        "iree-compile",
        f"{mlir_file}",
        "-o",
        f"{vmfb_file}",
    ] + extra_compiler_args

    if target == "host_cpu":
        exec_args += [
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        ]
    else:
        exec_args += [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={target}",
            "--iree-llvmgpu-enable-prefetch=true",
        ]

    ret_value, stdout, stderr = run_iree_command(exec_args)
    if ret_value == 0:
        if stderr:
            with open(dump_file, "w") as f:
                f.write(stderr.decode("utf-8"))
    else:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {mlir_file}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(stderr.decode("utf-8"))
        return mlir_file, None

    return mlir_file, vmfb_file
