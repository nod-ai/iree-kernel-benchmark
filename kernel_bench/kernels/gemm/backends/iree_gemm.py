import os
from typing import override
from iree.compiler import ir
from iree.compiler.dialects import arith, func, linalg, tensor

from kernel_bench.core.template import IREEKernelBenchmark
from kernel_bench.utils.device_utils import BenchDeviceContext
from kernel_bench.utils.iree_utils import (
    get_default_accumulator_element_type,
    get_default_result_element_type,
    run_iree_command,
)
from ..gemm_utils import GemmConfig


class IREEGemmBenchmark(IREEKernelBenchmark):
    config: GemmConfig

    @override
    def validate_config(self):
        if self.config.tA + self.config.tB == "TT":
            return False
        return True

    @override
    def compile_to_vmfb(self, mlir_path, vmfb_path):
        config = self.config

        with ir.Context():
            mlir_content = generate_iree_gemm_mlir(config, self.device_context)

        # Write MLIR content to file
        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        # Compile MLIR to VMFB
        exec_args = [
            "iree-compile",
            f"{mlir_path}",
            "-o",
            f"{vmfb_path}",
        ]

        if self.target == "host_cpu":
            exec_args += [
                "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
            ]
        else:
            exec_args += [
                "--iree-hal-target-backends=rocm",
                f"--iree-hip-target={self.target}",
                "--iree-llvmgpu-enable-prefetch=true",
            ]

        if self.path_config.dumps:
            dump_file = self.path_config.dump_for(
                "iree", config.get_name() + ".debug.mlir"
            )
            phase_dump = self.path_config.dumps / "iree" / config.get_name()
            exec_args.append(f"--dump-compilation-phases-to={phase_dump}")

        ret_value, stdout, stderr = run_iree_command(exec_args)
        if ret_value == 0:
            if stderr and self.path_config.dumps:
                with open(dump_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
        else:
            if self.path_config.dumps:
                error_file = self.path_config.dump_for(
                    "iree", "log", config.get_name() + "_error.txt"
                )
                self.logger.error(
                    f"Failed to compile {mlir_path}. Error dumped in {error_file}"
                )
                with open(error_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
            else:
                self.logger.error(
                    f"Failed to compile {mlir_path}. No dump directory specified."
                )

            return False

        return True


def convert_dtype_to_mlir(dtype: str) -> ir.Type:
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


def generate_iree_gemm_mlir(config: GemmConfig, device_context: BenchDeviceContext):
    K = config.K
    M = config.M
    N = config.N
    tA = config.tA
    tB = config.tB

    with ir.Location.name(config.get_name()):
        in_dtype = device_context.get_bench_dtype(config.dtype).to_full_string()
        operand_element_type = convert_dtype_to_mlir(in_dtype)
        acc_element_type = convert_dtype_to_mlir(
            get_default_accumulator_element_type(in_dtype)
        )
        result_element_type = convert_dtype_to_mlir(
            get_default_result_element_type(in_dtype, True)
        )
        is_integer = isinstance(operand_element_type, ir.IntegerType)
        literal_zero = (
            ir.IntegerAttr.get(acc_element_type, 0)
            if is_integer
            else ir.FloatAttr.get(acc_element_type, 0.0)
        )

        # Transpose A
        if tA == "T":
            arg0_type = ir.RankedTensorType.get([K, M], operand_element_type)
        else:
            arg0_type = ir.RankedTensorType.get([M, K], operand_element_type)
        # Transpose B
        if tB == "T":
            arg1_type = ir.RankedTensorType.get([N, K], operand_element_type)
        else:
            arg1_type = ir.RankedTensorType.get([K, N], operand_element_type)
        result_type = ir.RankedTensorType.get([M, N], result_element_type)

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(arg0_type, arg1_type)
            def main(arg0, arg1):
                zero_element = arith.constant(
                    value=literal_zero, result=acc_element_type
                )
                empty_tensor = tensor.empty(
                    element_type=acc_element_type,
                    sizes=[M, N],
                )
                filled_tensor = linalg.fill(zero_element, outs=[empty_tensor])

                # Define dimension expressions.
                d0 = ir.AffineDimExpr.get(0)  # M
                d1 = ir.AffineDimExpr.get(1)  # N
                d2 = ir.AffineDimExpr.get(2)  # K
                # Default maps.
                map_A = ir.AffineMap.get(3, 0, [d0, d2])
                map_B = ir.AffineMap.get(3, 0, [d2, d1])
                map_C = ir.AffineMap.get(3, 0, [d0, d1])
                if tA == "T":
                    map_A = ir.AffineMap.get(3, 0, [d2, d0])
                elif tB == "T":
                    map_B = ir.AffineMap.get(3, 0, [d1, d2])

                indexing_maps = ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(map_A),
                        ir.AffineMapAttr.get(map_B),
                        ir.AffineMapAttr.get(map_C),
                    ]
                )

                acc = linalg.matmul(
                    arg0,
                    arg1,
                    outs=[filled_tensor],
                    indexing_maps=indexing_maps,
                )

                if acc_element_type == result_element_type:
                    return acc
                if is_integer:
                    return arith.trunci(result_type, acc)
                return arith.truncf(result_type, acc)

        return f"{module}"
