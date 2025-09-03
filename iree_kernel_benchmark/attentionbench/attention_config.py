from dataclasses import dataclass
from typing import Union, Optional, Literal
import math
from abc import ABC, abstractmethod

from ..utils import dtype_to_bytes, OpConfig


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
        return f"{self.B}x{self.M}x{self.K1}x{self.dtype}"

    def get_key_shape(self) -> str:
        return f"{self.B}x{self.K2}x{self.K1}x{self.dtype}"

    def get_value_shape(self) -> str:
        return f"{self.B}x{self.K2}x{self.N}x{self.dtype}"

    def get_output_shape(self) -> str:
        return f"{self.B}x{self.M}x{self.N}x{self.dtype}"

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

    def get_shared_mem_bytes(self, spec):
        bytes_per_element = dtype_to_bytes(self.dtype)
        max_element_count = max(
            [
                spec.BLOCK_B * spec.BLOCK_M * self.K1,
                spec.BLOCK_B * spec.BLOCK_K2 * self.K1,
                spec.BLOCK_B * spec.BLOCK_K2 * spec.BLOCK_N,
                spec.BLOCK_B * spec.BLOCK_M * spec.BLOCK_N,
            ]
        )
        return max_element_count * bytes_per_element

    def get_flops(self) -> int:
        qk_matmul_flops = 2 * self.B * self.M * self.K2 * self.K1
        pv_matmul_flops = 2 * self.B * self.M * self.N * self.K2
        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops

    def get_runtime_args(self, backend_name):
        query_shape = self.get_query_shape()
        key_shape = self.get_key_shape()
        value_shape = self.get_value_shape()

        if backend_name.startswith("wave"):
            out_shape = self.get_output_shape()
            out_shape = "x".join(out_shape.split("x")[:-1] + ["f32"])
            inputs = [query_shape, key_shape, value_shape, out_shape]
            bench_function = "isolated_benchmark"
        else:
            inputs = [query_shape, key_shape, value_shape]
            bench_function = "main"

        return [f"--input={input}" for input in inputs] + [
            f"--function={bench_function}"
        ]


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

    def get_shared_mem_bytes(self, spec):
        bytes_per_element = dtype_to_bytes(self.dtype)
        max_element_count = max(
            [
                spec.BLOCK_B * spec.BLOCK_N_Q * spec.BLOCK_H * self.D_Q,
                spec.BLOCK_B * spec.BLOCK_N_KV * spec.BLOCK_H * self.D_Q,
                spec.BLOCK_B * spec.BLOCK_N_KV * spec.BLOCK_H * spec.BLOCK_D_KV,
                spec.BLOCK_B * spec.BLOCK_N_Q * spec.BLOCK_H * spec.BLOCK_D_KV,
            ]
        )
        return max_element_count * bytes_per_element

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
            out_shape = self.get_output_shape()
            out_shape = "x".join(out_shape.split("x")[:-1] + ["f32"])
            inputs = [query_shape, key_shape, value_shape, out_shape]
            bench_function = "isolated_benchmark"
        else:
            inputs = [query_shape, key_shape, value_shape]
            bench_function = "main"

        return [f"--input={input}" for input in inputs] + [
            f"--function={bench_function}"
        ]


@dataclass
class AttentionAttributes:
    """Unified attributes for all attention types"""

    attention_type: Literal["bmnk", "bshd"]
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    head_size_kv: int
    batch_size: Optional[int] = None
    dtype: str = "f16"
    # -----------------------
    # Prefill specific
    num_seqs: Optional[int] = None
    max_seq_len: Optional[int] = None
    total_seq_len: Optional[int] = None
    context_len: Optional[int] = None
    fixed_seq_len_prefix: Optional[int] = None
    fixed_seq_len_extend: Optional[int] = None
    # -----------------------
    # Vanilla attention
    query_seq_len: Optional[int] = None
    kv_seq_len: Optional[int] = None
    # -----------------------
    # Decode specific
    block_size: Optional[int] = None

    def get_name(self) -> str:
        if self.attention_type == "bmnk":
            return self.to_bmnk1k2().get_name()
        else:
            return self.to_bshd().get_name()

    def to_bmnk1k2(self) -> AttentionConfigBMNK:
        if self.batch_size is None:
            raise ValueError("batch_size is required for BMNK1K2 conversion")

        return AttentionConfigBMNK(
            B=self.num_query_heads,
            M=self.query_seq_len,
            N=self.head_size_kv,
            K1=self.head_size,
            K2=self.kv_seq_len,
            dtype=self.dtype,
        )

    def to_bshd(self) -> AttentionConfigBSHD:
        if self.num_seqs is None:
            raise ValueError("num_seqs is required for BSHD conversion")

        return AttentionConfigBSHD(
            B=self.num_seqs,
            H=self.num_query_heads,
            H_KV=self.num_kv_heads,
            N_Q=self.query_seq_len,
            D_KV=self.head_size_kv,
            D_Q=self.head_size,
            N_KV=self.kv_seq_len,
            dtype=self.dtype,
        )


def bmnk1k2_to_attention_attributes(
    config_bmnk: AttentionConfigBMNK,
) -> AttentionAttributes:
    return AttentionAttributes(
        attention_type="bmnk",
        num_query_heads=config_bmnk.B,
        num_kv_heads=config_bmnk.B,
        head_size=config_bmnk.K1,
        head_size_kv=config_bmnk.N,
        batch_size=1,
        query_seq_len=config_bmnk.M,
        kv_seq_len=config_bmnk.K2,
        dtype=config_bmnk.dtype,
    )


def bshd_to_attention_attributes(
    config_bshd: AttentionConfigBSHD,
) -> AttentionAttributes:
    return AttentionAttributes(
        attention_type="bshd",
        num_query_heads=config_bshd.H,
        num_kv_heads=config_bshd.H_KV,
        head_size=config_bshd.D_Q,
        head_size_kv=config_bshd.D_KV,
        num_seqs=config_bshd.B,
        query_seq_len=config_bshd.N_Q,
        kv_seq_len=config_bshd.N_KV,
        dtype=config_bshd.dtype,
    )
