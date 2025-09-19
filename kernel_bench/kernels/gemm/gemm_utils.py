from dataclasses import dataclass
from typing import Optional, Tuple, override

from kernel_bench.core.template import OpConfig
from iree.compiler import ir

from kernel_bench.utils.device_utils import dtype_to_bytes, get_device_specific_dtype

kDynamic = ir.ShapedType.get_dynamic_size()


@dataclass
class GemmConfig(OpConfig):
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

    def __post_init__(self):
        self.operand_element_type = get_device_specific_dtype(self.operand_element_type)
        self.accumulator_element_type = get_device_specific_dtype(
            self.accumulator_element_type
        )
        self.result_element_type = get_device_specific_dtype(self.result_element_type)

    @override
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

    @override
    def get_byte_count(self) -> int:
        operand_bytes_per_element = dtype_to_bytes(self.operand_element_type)
        result_bytes_per_element = dtype_to_bytes(self.result_element_type)
        M, N, K = self.get_runtime_dims()
        byte_count_input = (M + N) * K * operand_bytes_per_element
        byte_count_output = (M * N) * result_bytes_per_element
        return byte_count_input + byte_count_output

    def get_shared_mem_bytes(self, spec):
        return max(
            [
                spec.BLOCK_M * spec.BLOCK_K,
                spec.BLOCK_K * spec.BLOCK_N,
                spec.BLOCK_M * spec.BLOCK_N,
            ]
        )

    @override
    def get_flops(self) -> int:
        M, N, K = self.get_runtime_dims()
        flops = 2 * M * N * K
        return flops

    @override
    def to_dict(self):
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "tA": self.tA,
            "tB": self.tB,
            "dtype": self.operand_element_type,
        }

    def get_runtime_args(self, backend_name):
        inp1 = self.get_inp1()
        inp2 = self.get_inp2()
        out_shape = self.get_out()

        if backend_name == "wave":
            if "f8" in self.operand_element_type:
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
