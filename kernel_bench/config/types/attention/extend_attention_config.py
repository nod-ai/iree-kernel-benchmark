from dataclasses import dataclass
from typing import Any, Union, Optional
from dataclass_wizard import json_field
import torch
from pathlib import Path

from kernel_bench.config.types.attention.bshd_attention_config import (
    AttentionConfigBSHD,
    bshd_to_attention_attributes,
)
from kernel_bench.utils.dtypes import DeviceContext, dtype_to_bytes

from .base_attention_config import AttentionAttributes
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
    to_default_device,
)


@dataclass
class AttentionConfigExtend(AttentionConfigBSHD):
    attributes: AttentionAttributes = None
    inputs: "ExtendAttentionInputs" = json_field("inputs", default=None, dump=False)

    def __post_init__(self):
        if not self.attributes:
            self.attributes = bshd_to_attention_attributes(self)

    def get_name(self):
        return f"attention_extend_{self.B}x{self.H}x{self.N_Q}x{self.D_KV}x{self.dtype}"

    def get_byte_count(self):
        input_element_bytes = dtype_to_bytes(self.dtype)
        output_element_bytes = dtype_to_bytes("f32")

        q_size = self.N_Q * self.H * self.D_Q * input_element_bytes
        k_size = self.N_KV * self.H_KV * self.D_Q * input_element_bytes
        v_size = self.N_KV * self.H_KV * self.D_KV * input_element_bytes
        k_cache_size = self.N_KV * self.H_KV * self.D_Q * input_element_bytes
        v_cache_size = self.N_KV * self.H_KV * self.D_KV * input_element_bytes
        out_size = self.N_Q * self.H * self.D_KV * output_element_bytes

        return q_size + k_size + v_size + k_cache_size + v_cache_size + out_size

    def get_flops(self):
        # 4BHq​Sq​Sk​D = 4*B*48*Sq​*Sk​*128 = 24576*B*Sq*​Sk
        return 4 * self.B * self.H * self.N_Q * self.N_KV * self.D_Q

    def get_inputs(self, device_ctx: Optional["DeviceContext"] = None):
        """
        Get or create extend attention inputs.

        Args:
            device_ctx: Optional DeviceContext for dtype resolution. If not provided,
                       falls back to default torch.float16 for compatibility.
        """
        # Resolve dtype to torch dtype
        if device_ctx is not None:
            torch_dtype = device_ctx.dtype_to_torch(self.dtype)
        else:
            # Fallback for backward compatibility - assume f16
            torch_dtype = torch.float16

        self.inputs = create_extend_attention_inputs(self.attributes, torch_dtype)
        seq_len = self.inputs.max_len_extend
        self.attributes.max_seq_len = seq_len
        self.attributes.kv_seq_len = seq_len
        self.attributes.query_seq_len = seq_len
        self.N_Q = seq_len
        self.N_KV = seq_len
        return self.inputs


def validate_obj_attrs(obj: Any, attrs: list[str]):
    try:
        for attr in attrs:
            if not obj.__getattribute__(attr):
                raise Exception()
    except:
        raise ValueError(f"Could not find attribute {attr} in {obj}")


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
    q_extend: torch.Tensor
    k_extend: torch.Tensor
    v_extend: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    output: torch.Tensor
    max_len_extend: int
    logit_cap: float


def create_extend_attention_inputs(shape: AttentionAttributes, dtype=torch.float16):
    N_CTX = shape.context_len
    B = shape.num_seqs
    H_KV = shape.num_kv_heads
    H_Q = shape.num_query_heads
    D = shape.head_size
    b_seq_len_prefix = to_default_device(
        torch.ones((B,), dtype=torch.int32) * (N_CTX // 4)
    )
    if shape.fixed_seq_len_prefix:
        b_seq_len_prefix.fill_(shape.fixed_seq_len_prefix)
    b_seq_len_extend = to_default_device(
        torch.ones((B,), dtype=torch.int32) * (3 * N_CTX // 4)
    )
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

    # Create the inputs object
    inputs = ExtendAttentionInputs(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        output,
        max_len_extend,
        logit_cap,
    )

    return inputs
