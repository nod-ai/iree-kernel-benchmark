from ..utils import *
from dataclasses import dataclass

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
class ConvConfig(OpConfig):
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

    def get_runtime_args(self, backend_name):
        image_shape = self.get_img_shape()
        filter_shape = self.get_kernel_shape()

        runtime_args = [
            f"--input={image_shape}",
            f"--input={filter_shape}",
        ]

        if backend_name == "wave":
            out_shape = self.get_out_shape()
            runtime_args += [f"--input={out_shape}", "--function=isolated_benchmark"]
        else:
            runtime_args.append("--function=main")

        return runtime_args


@dataclass
class ConvTuningSpec(TuningSpec):
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    ELEMS_PER_THREAD: int
