"""
GEMM (General Matrix Multiply) configuration types.

This module defines configuration dataclasses for GEMM operations.
"""

from dataclasses import dataclass
from typing import override

from kernel_bench.config.base import OpConfig
from kernel_bench.utils.dtypes import dtype_to_bytes


@dataclass
class GemmConfig(OpConfig):
    """
    Configuration for General Matrix Multiplication (GEMM) operations.

    Represents: C = A @ B where A is MxK, B is KxN, C is MxN

    Attributes:
        M: Number of rows in matrix A and output matrix C
        N: Number of columns in matrix B and output matrix C
        K: Number of columns in matrix A and rows in matrix B
        tA: Transpose flag for matrix A ("T" for transpose, "N" for normal)
        tB: Transpose flag for matrix B ("T" for transpose, "N" for normal)
        dtype: Data type (e.g., "f16", "f32", "bf16", "f8E4M3FNUZ")
    """

    M: int
    N: int
    K: int
    tA: str
    tB: str
    dtype: str

    @override
    def get_name(self):
        return f"gemm_{self.M}_{self.N}_{self.K}_{self.dtype}_{self.tA}{self.tB}"

    @override
    def get_byte_count(self):
        operand_bytes_per_element = dtype_to_bytes(self.dtype)
        result_bytes_per_element = dtype_to_bytes("f32")
        byte_count_input = (self.M + self.N) * self.K * operand_bytes_per_element
        byte_count_output = (self.M * self.N) * result_bytes_per_element
        return byte_count_input + byte_count_output

    @override
    def get_flops(self):
        return 2 * self.M * self.N * self.K

    @override
    def to_dict(self):
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "transpose": self.tA + self.tB,
            "dtype": self.dtype,
        }

    @override
    @classmethod
    def from_dict(cls, obj) -> "GemmConfig":
        """
        Convert dictionary to config instance.
        """
        return GemmConfig(
            M=obj["M"],
            N=obj["N"],
            K=obj["K"],
            tA=obj["transpose"][0],
            tB=obj["transpose"][1],
            dtype=obj["dtype"],
        )
