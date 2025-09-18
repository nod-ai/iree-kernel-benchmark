import os
from typing import override
from iree.compiler import ir
from iree.compiler.dialects import arith, func, linalg, tensor

from kernel_bench.core.template import IREEKernelBenchmark
from kernel_bench.utils.iree_utils import run_iree_command
from ..gemm_utils import GemmConfig, kDynamic


class IREEGemmBenchmark(IREEKernelBenchmark):
    config: GemmConfig

    def _convert_dtype_to_mlir(self, dtype: str) -> ir.Type:
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

    def _generate_mlir(self, config: GemmConfig):
        K = config.K
        M = config.M
        N = config.N
        tA = config.tA
        tB = config.tB

        with ir.Location.name(config.get_name()):
            operand_element_type = self._convert_dtype_to_mlir(
                config.operand_element_type
            )
            acc_element_type = self._convert_dtype_to_mlir(
                config.accumulator_element_type
            )
            result_element_type = self._convert_dtype_to_mlir(
                config.result_element_type
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
                arg0_M_idx = 1
            else:
                arg0_type = ir.RankedTensorType.get([M, K], operand_element_type)
                arg0_M_idx = 0
            # Transpose B
            if tB == "T":
                arg1_type = ir.RankedTensorType.get([N, K], operand_element_type)
                arg1_N_idx = 0
            else:
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

    @override
    def compile_to_vmfb(self, mlir_path, vmfb_path):
        config = self.config

        with ir.Context():
            mlir_content = self._generate_mlir(config)

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

        if self.dump_dir:
            os.makedirs(self.dump_dir / "iree", exist_ok=True)
            dump_file = self.dump_dir / "iree" / (config.get_name() + ".debug.mlir")
            phase_dump = self.dump_dir / "iree" / config.get_name()
            exec_args.append(f"--dump-compilation-phases-to={phase_dump}")

        ret_value, stdout, stderr = run_iree_command(exec_args)
        if ret_value == 0:
            if stderr and self.dump_dir:
                with open(dump_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
        else:
            if self.dump_dir:
                error_file = (
                    self.dump_dir / "iree" / "log" / (config.get_name() + "_error.txt")
                )
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
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
