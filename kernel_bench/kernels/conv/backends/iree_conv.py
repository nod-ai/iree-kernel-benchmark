import os
from kernel_bench.core.template import IREEKernelBenchmark
from kernel_bench.utils.iree_utils import run_iree_command
from ..conv_utils import ConvConfig

# MLIR template strings for convolution operations
FUNC_ARGS = r"""%arg0: tensor<{LHS_TYPE}>, %arg1: tensor<{RHS_TYPE}>"""
CONSTANTS = r"""
    %arg0 = util.unfoldable_constant dense<{ONE}> : tensor<{LHS_TYPE}>
    %arg1 = util.unfoldable_constant dense<{ONE}> : tensor<{RHS_TYPE}>"""

CONV = r"""%11 = linalg.conv_2d_{CONV_TYPE} {{dilations = dense<1> : vector<2xi64>, strides = dense<{STRIDE}> : vector<2xi64>}} ins(%arg0, %arg1 : tensor<{INPUT_TYPE}>, tensor<{FILTER_TYPE}>) outs(%10 : tensor<{OUTPUT_TYPE}>) -> tensor<{OUTPUT_TYPE}>"""

CONV_Q = r"""%c0_i32 = arith.constant 0 : i32
    %11 = linalg.conv_2d_{CONV_TYPE}_q {{dilations = dense<1> : vector<2xi64>, strides = dense<{STRIDE}> : vector<2xi64>}} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<{INPUT_TYPE}>, tensor<{FILTER_TYPE}>, i32, i32) outs(%10 : tensor<{OUTPUT_TYPE}>) -> tensor<{OUTPUT_TYPE}>"""

TEST = r"""util.func public @{FUNC_NAME}({FUNC_ARGS}) -> tensor<{OUT_TYPE}> {{{CONSTANT_INPUTS}
    %cst = arith.constant {ZERO} : {OUT_ELEM_TYPE}
    %9 = tensor.empty() : tensor<{OUT_TYPE}>
    %10 = linalg.fill ins(%cst : {OUT_ELEM_TYPE}) outs(%9 : tensor<{OUT_TYPE}>) -> tensor<{OUT_TYPE}>
    {OPERATION}
    util.return %11 : tensor<{OUT_TYPE}>
}}
"""


class IREEConvBenchmark(IREEKernelBenchmark):
    config: ConvConfig

    def _generate_mlir(self, config: ConvConfig):
        n = config.N
        h = config.H
        w = config.W
        c = config.C
        p = config.P
        q = config.Q
        f = config.F
        stride = config.S
        operation = config.OP

        in_dtype = self.device_context.get_bench_dtype(
            config.input_dtype
        ).to_full_string()
        out_dtype = self.device_context.get_bench_dtype(
            config.output_dtype
        ).to_full_string()

        dtypes = f"{in_dtype}x{in_dtype}x{out_dtype}"
        elem_types = dtypes.split("x")
        in_h = str(int(h) * int(stride) + int(p) - 1)
        in_w = str(int(w) * int(stride) + int(q) - 1)
        if "nhwc" in operation:
            conv_type = "nhwc_hwcf"
            lhs = f"{n}x{in_h}x{in_w}x{c}x{elem_types[0]}"
            rhs = f"{p}x{q}x{c}x{f}x{elem_types[1]}"
            out = f"{n}x{h}x{w}x{f}x{elem_types[2]}"
        if "nchw" in operation:
            conv_type = "nchw_fchw"
            lhs = f"{n}x{c}x{in_h}x{in_w}x{elem_types[0]}"
            rhs = f"{f}x{c}x{p}x{q}x{elem_types[1]}"
            out = f"{n}x{f}x{h}x{w}x{elem_types[2]}"
        one = "1"
        zero = "0"
        if elem_types[0][0] == "f" or elem_types[0][0] == "b":
            one = "1.0"
            zero = "0.0"
        conv_template = CONV
        if "q" in operation:
            conv_template = CONV_Q
        operation = conv_template.format(
            INPUT_TYPE=lhs,
            FILTER_TYPE=rhs,
            OUTPUT_TYPE=out,
            CONV_TYPE=conv_type,
            STRIDE=stride,
        )

        constants = ""
        func_args = ""
        func_args = FUNC_ARGS.format(
            LHS_TYPE=lhs,
            RHS_TYPE=rhs,
        )

        mlir = TEST.format(
            FUNC_NAME="main",
            FUNC_ARGS=func_args,
            CONSTANT_INPUTS=constants,
            LHS_TYPE=lhs,
            RHS_TYPE=rhs,
            OUT_TYPE=out,
            OUT_ELEM_TYPE=elem_types[2],
            ZERO=zero,
            OPERATION=operation,
        )
        return mlir

    def compile_to_vmfb(self, mlir_path, vmfb_path):
        config = self.config

        # Name with tag is used for filenames so that duplicate configs with
        # different tags will not clobber eachother.
        if self.path_config.dumps:
            dump_file = self.path_config.dump_for(
                "iree", config.get_name() + ".stderr.mlir"
            )

        # Generate mlir content
        mlir_content = self._generate_mlir(config)

        # Write MLIR content to file
        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        # TODO: Do not hardcode device information, instead pass it as a class
        # Compile MLIR to vmfb
        exec_args = [
            "iree-compile",
            # Input file
            f"{mlir_path}",
            # Output file
            "-o",
            f"{vmfb_path}",
            # Target Device: hip
            "--iree-hal-target-device=hip",
            # Device: MI300x
            f"--iree-hip-target={self.target}",
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
