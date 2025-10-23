"""
Convolution configuration types.

This module defines configuration dataclasses for convolution operations.
"""

from dataclasses import dataclass
from typing import override

from kernel_bench.config.base import OpConfig
from kernel_bench.utils.dtypes import dtype_to_bytes


@dataclass
class ConvConfig(OpConfig):
    """
    Configuration for 2D Convolution operations.

    Attributes:
        N: Batch size
        H: Output height
        W: Output width
        C: Input channels
        P: Kernel height
        Q: Kernel width
        F: Output channels (number of filters)
        S: Stride
        OP: Operation type (e.g., "conv_2d_nhwc_hwcf", "conv_2d_nchw_fchw")
        input_dtype: Input data type
        output_dtype: Output data type
    """

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

    @override
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

    @override
    def get_byte_count(self) -> int:
        bytes_per_input = dtype_to_bytes(self.input_dtype)
        bytes_per_output = dtype_to_bytes(self.output_dtype)
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

    @override
    def get_flops(self) -> int:
        batch = self.N
        output_channels = self.F
        output_width = self.W
        output_height = self.H
        k_width = self.Q
        k_height = self.P
        input_channels = self.C
        operation_per_pixel = k_width * k_height * input_channels * 2
        output_pixels_per_batch = output_width * output_height * output_channels
        flops = operation_per_pixel * output_pixels_per_batch * batch
        return flops

    @override
    def to_dict(self):
        return {
            "N": self.N,
            "H": self.H,
            "W": self.W,
            "C": self.C,
            "P": self.P,
            "Q": self.Q,
            "F": self.F,
            "S": self.S,
            "OP": self.OP,
            "dtype": self.input_dtype,
        }

    @override
    @classmethod
    def from_dict(cls, obj):
        return ConvConfig(
            N=obj["N"],
            H=obj["H"],
            W=obj["W"],
            C=obj["C"],
            P=obj["P"],
            Q=obj["Q"],
            F=obj["F"],
            S=obj["S"],
            OP=obj["OP"],
            input_dtype=obj["dtype"],
            output_dtype="f32",
        )

    def decode_op(self) -> tuple[str, str]:
        """Decode operation string into base op and variant."""
        if self.OP.startswith("conv_2d_"):
            return "conv_2d", self.OP[len("conv_2d_") :]
        raise ValueError(f"Unsupported op: {self.OP}")
