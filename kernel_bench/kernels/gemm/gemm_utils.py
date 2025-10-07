from dataclasses import dataclass
from typing import override

from kernel_bench.core.template import OpConfig
from kernel_bench.utils.device_utils import dtype_to_bytes, stringify_shape


@dataclass
class GemmConfig(OpConfig):
    M: int
    N: int
    K: int
    tA: str
    tB: str
    dtype: str

    @override
    def get_name(self):
        return f"gemm_{self.M}_{self.N}_{self.K}_{self.dtype}_{self.tA}{self.tB}"

    def get_inp1(self) -> str:
        if self.tA == "T":
            return stringify_shape((self.K, self.M), self.dtype)
        else:
            return stringify_shape((self.M, self.K), self.dtype)

    def get_inp2(self) -> str:
        if self.tB == "T":
            return stringify_shape((self.N, self.K), self.dtype)
        else:
            return stringify_shape((self.K, self.N), self.dtype)

    def get_out(self) -> str:
        return stringify_shape((self.M, self.N), self.dtype)

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
    def get_runtime_args(self, backend_name):
        inp1 = self.get_inp1()
        inp2 = self.get_inp2()
        out_shape = self.get_out()

        if backend_name == "wave":
            if "f8" in self.dtype:
                inp1 = "x".join(inp1.split("x")[:-1] + ["f16"])
                inp2 = "x".join(inp2.split("x")[:-1] + ["f16"])
            out_shape = "x".join(out_shape.split("x")[:-1] + ["f32"])

            runtime_args = [
                f"--input={inp1}",
                f"--input={inp2}",
                f"--input={out_shape}",
                "--function=isolated_benchmark",
            ]
        else:
            runtime_args = [
                f"--input={inp1}",
                f"--input={inp2}",
                "--function=main",
            ]

        return runtime_args
