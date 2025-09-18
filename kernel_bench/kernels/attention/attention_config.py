from dataclasses import dataclass
from typing import Any, Union, Optional, Literal
import torch
from abc import ABC, abstractmethod

from kernel_bench.utils.bench_utils import OpConfig, change_shape_dtype
from kernel_bench.utils.device_utils import (
    dtype_to_bytes,
    dtype_to_torch,
    stringify_shape,
    stringify_tensor_shape,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
)


@dataclass
class AttentionAttributes:
    """Unified attributes for all attention types"""

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


@dataclass
class AttentionConfigBMNK(OpConfig):
    B: int
    M: int
    N: int
    K1: int
    K2: int
    dtype: str
    attributes: AttentionAttributes = None

    def __post_init__(self):
        if not self.attributes:
            self.attributes = bmnk1k2_to_attention_attributes(self)

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
    attributes: AttentionAttributes

    def __post_init__(self):
        if not self.attributes:
            self.attributes = bshd_to_attention_attributes(self)

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


class AttentionConfigExtend(AttentionConfigBSHD):
    inputs: "ExtendAttentionInputs" = None

    def __post_init__(self):
        super().__post_init__()
        self.inputs = create_extend_attention_inputs(
            self.attributes, dtype_to_torch(self.dtype)
        )
        seq_len = self.inputs.max_len_extend
        self.attributes.max_seq_len = seq_len
        self.attributes.kv_seq_len = seq_len
        self.attributes.query_seq_len = seq_len
        self.N_Q = seq_len
        self.N_KV = seq_len

    def get_runtime_args(self, backend_name):
        bench_inputs = [
            stringify_shape(self.inputs.q_extend_shape, self.dtype),
            stringify_shape(self.inputs.k_extend_shape, self.dtype),
            stringify_shape(self.inputs.v_extend_shape, self.dtype),
            stringify_shape(self.inputs.k_buffer_shape, self.dtype),
            stringify_shape(self.inputs.v_buffer_shape, self.dtype),
            stringify_shape(self.inputs.qo_indptr_shape, "i32"),
            stringify_shape(self.inputs.kv_indptr_shape, "i32"),
            stringify_shape(self.inputs.kv_indices_shape, "i32"),
            stringify_shape(self.inputs.output_shape, "f32"),
            stringify_shape(self.inputs.max_len_extend, "i32"),
        ]
        print(bench_inputs)
        bench_function = "isolated_benchmark" if backend_name == "wave" else "main"
        return [f"--input={input}" for input in bench_inputs] + [
            f"--function={bench_function}"
        ]


def bmnk1k2_to_attention_attributes(
    config_bmnk: AttentionConfigBMNK,
) -> AttentionAttributes:
    if config_bmnk.attributes:
        return config_bmnk.attributes
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


def bshd_to_attention_attributes(
    config_bshd: AttentionConfigBSHD,
) -> AttentionAttributes:
    if config_bshd.attributes:
        return config_bshd.attributes
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


def validate_obj_attrs(obj: Any, attrs: list[str]):
    try:
        for attr in attrs:
            if not obj.__getattribute__(attr):
                raise Exception()
    except:
        raise ValueError(f"Could not find attribute {attr} in {obj}")


def attention_attributes_to_bmnk1k2(
    shape: AttentionAttributes,
) -> AttentionConfigBMNK:
    validate_obj_attrs(
        shape,
        [
            "num_query_heads",
            "query_seq_len",
            "head_size_kv",
            "head_size",
            "kv_seq_len",
            "dtype",
        ],
    )
    return AttentionConfigBMNK(
        B=shape.num_query_heads,
        M=shape.query_seq_len,
        N=shape.head_size_kv,
        K1=shape.head_size,
        K2=shape.kv_seq_len,
        dtype=shape.dtype,
        attributes=shape,
    )


def attention_attributes_to_bshd(
    shape: AttentionAttributes,
) -> AttentionConfigBSHD:
    validate_obj_attrs(
        shape,
        [
            "num_seqs",
            "num_query_heads",
            "num_kv_heads",
            "query_seq_len",
            "head_size_kv",
            "head_size",
            "kv_seq_len",
            "dtype",
        ],
    )
    return AttentionConfigBSHD(
        B=shape.num_seqs,
        H=shape.num_query_heads,
        H_KV=shape.num_kv_heads,
        N_Q=shape.query_seq_len,
        D_KV=shape.head_size_kv,
        D_Q=shape.head_size,
        N_KV=shape.kv_seq_len,
        dtype=shape.dtype,
        attributes=shape,
    )


def attention_attributes_to_extend(
    shape: AttentionAttributes,
) -> AttentionConfigExtend:
    validate_obj_attrs(
        shape,
        [
            "num_seqs",
            "num_query_heads",
            "num_kv_heads",
            "head_size_kv",
            "head_size",
            "dtype",
        ],
    )
    return AttentionConfigExtend(
        B=shape.num_seqs,
        H=shape.num_query_heads,
        H_KV=shape.num_kv_heads,
        N_Q=shape.query_seq_len,
        D_KV=shape.head_size_kv,
        D_Q=shape.head_size,
        N_KV=shape.kv_seq_len,
        dtype=shape.dtype,
        attributes=shape,
    )


@dataclass
class ExtendAttentionInputs:
    q_extend_shape: torch.Size
    k_extend_shape: torch.Size
    v_extend_shape: torch.Size
    k_buffer_shape: torch.Size
    v_buffer_shape: torch.Size
    qo_indptr_shape: torch.Size
    kv_indptr_shape: torch.Size
    kv_indices_shape: torch.Size
    output_shape: torch.Size
    max_len_extend: int
    logit_cap: float


def create_extend_attention_inputs(
    shape: AttentionAttributes,
    dtype=torch.float16,
):
    N_CTX = shape.context_len
    B = shape.num_seqs
    H_KV = shape.num_kv_heads
    H_Q = shape.num_query_heads
    D = shape.head_size
    b_seq_len_prefix = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_prefix:
        b_seq_len_prefix.fill_(shape.fixed_seq_len_prefix)
    b_seq_len_extend = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_extend:
        b_seq_len_extend.fill_(shape.fixed_seq_len_extend)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_req_idx = device_arange(B, dtype=torch.int32)
    b_start_loc = device_zeros((B,), dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = device_zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = device_zeros((B + 1,), dtype=torch.int32)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = device_zeros((b_seq_len_prefix.sum().item(),), dtype=torch.int32)

    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
        )
    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )
    v_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )

    k_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    q_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = device_empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype
        ).normal_(mean=0.1, std=0.2)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = device_zeros((B + 1,), dtype=torch.int32)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)
    logit_cap = 30.0

    b_seq_mask_len = b_seq_len_extend * b_seq_len
    # NOTE: Custom mask is of causal nature in this test. Random mask numerics
    # is not tested.
    custom_mask = device_full(
        (b_seq_mask_len.sum().item(),), fill_value=1, dtype=torch.int8
    )
    mask_offsets = device_zeros((B + 1,), dtype=torch.int32)
    mask_offsets[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
    for i in range(B):
        causal_mask = (
            torch.tril(
                device_full(
                    (b_seq_len_extend[i], b_seq_len_extend[i]),
                    fill_value=1,
                    dtype=torch.int8,
                ),
                diagonal=0,
            )
            == 1
        )
        prefix_mask = device_full(
            (b_seq_len_extend[i], b_seq_len_prefix[i]), fill_value=1, dtype=torch.int8
        )
        mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
        custom_mask[mask_offsets[i] : mask_offsets[i + 1]] = mask_flatten

    max_rpe_context_length = 10
    rpe_bias = device_zeros(max_rpe_context_length + 1, dtype=torch.float32)
    rpe_bias.copy_(device_randn(max_rpe_context_length + 1, dtype=torch.float32))
    rpe_bias[max_rpe_context_length] = 0

    output = device_zeros(
        extend_token_num, shape.num_query_heads, shape.head_size, dtype=torch.float32
    )

    return ExtendAttentionInputs(
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        k_buffer.shape,
        v_buffer.shape,
        qo_indptr.shape,
        kv_indptr.shape,
        kv_indices.shape,
        output.shape,
        max_len_extend,
        logit_cap,
    )
