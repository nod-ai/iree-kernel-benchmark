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
class AttentionConfigBSHD(OpConfig):
    B: int  # num_seqs
    H: int  # num_query_heads
    H_KV: int  # num_kv_heads
    N_Q: int  # query_seq_len
    D_KV: int  # head_size_kv
    D_Q: int  # head_size
    N_KV: int  # kv_seq_len
    dtype: str

    def get_name(self) -> str:
        return f"attention_bshd_{self.B}x{self.H}x{self.H_KV}x{self.N_Q}x{self.D_KV}x{self.D_Q}x{self.N_KV}x{self.dtype}"

    def get_query_shape(self) -> str:
        return f"{self.B}x{self.N_Q}x{self.H}x{self.D_Q}x{self.dtype}"

    def get_key_shape(self) -> str:
        return f"{self.B}x{self.N_KV}x{self.H_KV}x{self.D_Q}x{self.dtype}"

    def get_value_shape(self) -> str:
        return f"{self.B}x{self.N_KV}x{self.H_KV}x{self.D_KV}x{self.dtype}"

    def get_output_shape(self) -> str:
        return f"{self.B}x{self.N_Q}x{self.H}x{self.D_KV}x{self.dtype}"

    def get_byte_count(self) -> int:
        bytes_per_element = dtype_to_bytes(self.dtype)
        element_count = (
            (self.B * self.N_Q * self.H * self.D_Q)  # Query
            + (self.B * self.N_KV * self.H_KV * self.D_Q)  # Key
            + (self.B * self.N_KV * self.H_KV * self.D_KV)  # Value
            + (self.B * self.N_Q * self.H * self.D_KV)  # Output
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        # QK matmul: (B, N_Q, H, D_Q) x (B, N_KV, H_KV, D_Q) -> (B, H, N_Q, N_KV)
        # Assuming H_KV is broadcast to H for computation
        qk_matmul_flops = 2 * self.B * self.H * self.N_Q * self.N_KV * self.D_Q

        # PV matmul: (B, H, N_Q, N_KV) x (B, N_KV, H_KV, D_KV) -> (B, N_Q, H, D_KV)
        # Assuming H_KV is broadcast to H for computation
        pv_matmul_flops = 2 * self.B * self.H * self.N_Q * self.N_KV * self.D_KV

        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops

    def get_runtime_args(self, backend_name):
        query_shape = self.get_query_shape()
        key_shape = self.get_key_shape()
        value_shape = self.get_value_shape()

        if backend_name.startswith("wave"):
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
            "H": self.H,
            "H_KV": self.H_KV,
            "N_Q": self.N_Q,
            "D_KV": self.D_KV,
            "D_Q": self.D_Q,
            "N_KV": self.N_KV,
            "dtype": self.dtype,
        }


def bshd_to_attention_attributes(
    config_bshd: AttentionConfigBSHD,
) -> AttentionAttributes:
    return AttentionAttributes(
        num_query_heads=config_bshd.H,
        num_kv_heads=config_bshd.H_KV,
        head_size=config_bshd.D_Q,
        head_size_kv=config_bshd.D_KV,
        num_seqs=config_bshd.B,
        max_seq_len=max(config_bshd.N_Q, config_bshd.N_KV),
        query_seq_len=config_bshd.N_Q,
        kv_seq_len=config_bshd.N_KV,
        dtype=config_bshd.dtype,
    )
