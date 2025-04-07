from ..utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class ConvConfig:
    N: int
    H: int
    W: int
    C: int
    P: int
    Q: int
    F: int
    S: int
    OP: str
    input_dtype: str
    output_dtype: str

    def get_name(self) -> str:
        return (
            self.OP
            + "_"
            + f"{self.N}x{self.H}x{self.W}x{self.C}x{self.P}x{self.Q}x{self.F}"
            + "_"
            + f"{self.input_dtype}x{self.input_dtype}x{self.output_dtype}"
            + "_stride"
            + str(self.S)
        )

    def __eq__(self, other):
        if not isinstance(other, ConvConfig):
            return NotImplemented
        return self.get_name() == other.get_name()

    def __hash__(self):
        return hash(self.get_name())

    def get_img_shape(self) -> str:
        if "nhwc" in self.OP:
            in_h = self.H * self.S + self.P - 1
            in_w = self.W * self.S + self.Q - 1
            return f"{self.N}x{in_h}x{in_w}x{self.C}x{self.input_dtype}"
        if "nchw" in self.OP:
            in_h = self.H * self.S + self.P - 1
            in_w = self.W * self.S + self.Q - 1
            return f"{self.N}x{self.C}x{in_h}x{in_w}x{self.input_dtype}"

    def get_kernel_shape(self) -> str:
        if "nhwc" in self.OP:
            return f"{self.P}x{self.Q}x{self.C}x{self.F}x{self.input_dtype}"
        if "nchw" in self.OP:
            return f"{self.F}x{self.C}x{self.P}x{self.Q}x{self.input_dtype}"

    def get_out_shape(self) -> str:
        padding = 0
        in_h = self.H * self.S + self.P - 1
        in_w = self.W * self.S + self.Q - 1
        h_out = (in_h + 2 * padding - self.P) // self.S + 1
        w_out = (in_w + 2 * padding - self.Q) // self.S + 1
        n = self.N
        nf = self.F
        if "nhwc" in self.OP:
            return f"{n}x{h_out}x{w_out}x{nf}x{self.output_dtype}"
        if "nchw" in self.OP:
            return f"{n}x{nf}x{h_out}x{w_out}x{self.output_dtype}"

    def get_byte_count(self) -> int:
        dtype_bits_map = {
            "f32": 32,
            "f16": 16,
            "bf16": 16,
            "f8E4M3FNUZ": 8,
            "i8": 8,
            "i32": 32,
        }
        bytes_per_input = dtype_bits_map[self.input_dtype] // 8
        bytes_per_output = dtype_bits_map[self.output_dtype] // 8
        batch = self.N
        in_h = self.H * self.S + self.P - 1
        in_w = self.W * self.S + self.Q - 1
        input_channels = self.C
        output_channels = self.F
        output_width = self.W
        output_height = self.H
        k_width = self.Q
        k_height = self.P
        byte_count = (
            (batch * input_channels * in_w * in_h * bytes_per_input)
            + (
                batch
                * output_channels
                * output_width
                * output_height
                * bytes_per_output
            )
            + (k_width * k_height * input_channels * output_channels * bytes_per_input)
        )
        return byte_count

    def get_flops(self) -> int:
        batch = self.N
        in_h = self.H * self.S + self.P - 1
        in_w = self.W * self.S + self.Q - 1
        input_channels = self.C
        output_channels = self.F
        output_width = self.W
        output_height = self.H
        k_width = self.Q
        k_height = self.P
        operation_per_pixel = k_width * k_height * input_channels * 2
        output_pixels_per_batch = output_width * output_height * output_channels
        flops = operation_per_pixel * output_pixels_per_batch * batch
        return flops


def generate_mlir(config: ConvConfig):
    n = config.N
    h = config.H
    w = config.W
    c = config.C
    p = config.P
    q = config.Q
    f = config.F
    stride = config.S
    operation = config.OP
    dtypes = f"{config.input_dtype}x{config.input_dtype}x{config.output_dtype}"
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


def compile_conv_config(
    tag: str,
    config: ConvConfig,
    kernel_dir: Path,
    vmfb_dir: Path,
    extra_compiler_args: list[str],
) -> tuple[Path, Optional[Path]]:
    # Name with tag is used for filenames so that duplicate configs with
    # different tags will not clobber eachother.
    name_with_tag = tag + "-" + config.get_name()
    mlir_file = kernel_dir / (name_with_tag + ".mlir")
    vmfb_file = vmfb_dir / (name_with_tag + ".vmfb")
    dump_file = kernel_dir / (name_with_tag + ".stderr.mlir")
    files_path = vmfb_dir / name_with_tag

    # Generate mlir content
    mlir_content = generate_mlir(config)

    # Write MLIR content to file
    with open(mlir_file, "w") as f:
        f.write(mlir_content)

    # TODO: Do not hardcode device information, instead pass it as a class
    # Compile MLIR to vmfb
    exec_args = [
        "iree-compile",
        # Input file
        f"{mlir_file}",
        # Output file
        "-o",
        f"{vmfb_file}",
        # Target Device: hip
        "--iree-hal-target-device=hip",
        # Device: MI300x
        "--iree-hip-target=gfx942",
        f"--iree-hal-dump-executable-files-to={files_path}",
    ] + extra_compiler_args

    print(" ".join(exec_args))

    ret_value, stdout, stderr = run_iree_command(exec_args)
    if ret_value == 0:
        print(f"Successfully compiled {mlir_file} to {vmfb_file}")
        if stderr:
            with open(dump_file, "w") as f:
                f.write(stderr.decode("utf-8"))
    else:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {mlir_file}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(stderr.decode("utf-8"))
        return mlir_file, None, None

    return mlir_file, vmfb_file, files_path
