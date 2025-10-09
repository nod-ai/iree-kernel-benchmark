from dataclasses import dataclass
from kernel_bench.config.base import OpConfig
from kernel_bench.utils.bench_utils import OpConfig, change_shape_dtype
from kernel_bench.utils.device_utils import (
    dtype_to_bytes,
    dtype_to_torch,
    stringify_shape,
    stringify_tensor_shape,
)
from .base_attention_config import AttentionAttributes


@dataclass
class AttentionConfigBMNK(OpConfig):
    B: int
    M: int
    N: int
    K1: int
    K2: int
    dtype: str

    def get_name(self) -> str:
        return f"attention_bmnk1k2_{self.B}x{self.M}x{self.N}x{self.K1}x{self.K2}x{self.dtype}"

    def get_query_shape(self) -> str:
        return stringify_shape((self.B, self.M, self.K1), self.dtype)

    def get_key_shape(self) -> str:
        return stringify_shape((self.B, self.K2, self.K1), self.dtype)

    def get_value_shape(self) -> str:
        return stringify_shape((self.B, self.K2, self.N), self.dtype)

    def get_output_shape(self) -> str:
        return stringify_shape((self.B, self.M, self.N), self.dtype)

    def get_byte_count(self) -> int:
        bytes_per_element = dtype_to_bytes(self.dtype)
        element_count = (
            (self.B * self.M * self.K1)
            + (self.B * self.K2 * self.K1)
            + (self.B * self.K2 * self.N)
            + (self.B * self.M * self.N)
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        qk_matmul_flops = 2 * self.B * self.M * self.K2 * self.K1
        pv_matmul_flops = 2 * self.B * self.M * self.N * self.K2
        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops

    def get_runtime_args(self, backend_name):
        query_shape = self.get_query_shape()
        key_shape = self.get_key_shape()
        value_shape = self.get_value_shape()

        if backend_name == "wave":
            inputs = [query_shape, key_shape, value_shape]
            if "f8" in self.dtype:
                inputs = [change_shape_dtype(shape, "f16") for shape in inputs]
            out_shape = change_shape_dtype(self.get_output_shape(), "f32")
            inputs.append(out_shape)
            bench_function = "isolated_benchmark"

        else:
            inputs = [query_shape, key_shape, value_shape]
            bench_function = "main"

        return [f"--input={input}" for input in inputs] + [
            f"--function={bench_function}"
        ]

    def to_dict(self):
        return {
            "B": self.B,
            "M": self.M,
            "N": self.N,
            "K1": self.K1,
            "K2": self.K2,
            "dtype": self.dtype,
        }


def bmnk1k2_to_attention_attributes(
    config_bmnk: AttentionConfigBMNK,
) -> AttentionAttributes:
    return AttentionAttributes(
        num_query_heads=config_bmnk.B,
        num_kv_heads=config_bmnk.B,
        head_size=config_bmnk.K1,
        head_size_kv=config_bmnk.N,
        batch_size=1,
        query_seq_len=config_bmnk.M,
        kv_seq_len=config_bmnk.K2,
        dtype=config_bmnk.dtype,
    )
