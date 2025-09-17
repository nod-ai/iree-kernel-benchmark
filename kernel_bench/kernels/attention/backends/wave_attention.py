from dataclasses import replace
import torch
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel
from kernel_bench.tuning.hyperparam import (
    TuningParameter,
    CategoricalBounds,
    IntegerBounds,
)
from kernel_bench.core.template import WaveKernel, WaveKernelBenchmark
from kernel_bench.utils.device_utils import dtype_to_bytes, dtype_to_torch
from wave_lang.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from ..attention_config import (
    AttentionAttributes,
    AttentionConfigBMNK,
    AttentionConfigBSHD,
    bmnk1k2_to_attention_attributes,
    bshd_to_attention_attributes,
)
from typing import override

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.kernel.wave.templates.gqa_vanilla_attention import (
    get_gqa_bshd_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
)


class WaveAttentionMHABenchmark(WaveKernelBenchmark):
    config: AttentionConfigBMNK

    def setup_parameters(self):
        config = self.config

        if "f8" in config.dtype:
            mfma_bounds = CategoricalBounds(
                [
                    (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
                    (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
                ]
            )
        else:
            mfma_bounds = CategoricalBounds(
                [
                    (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                    (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
                ]
            )

        self.mfma_variant = self.add_param(
            "mfma_variant", mfma_bounds, initial_value=0, include_hyperparam=False
        )
        self.BLOCK_B = self.add_param("BLOCK_B", IntegerBounds(min=1, max=config.B))
        self.BLOCK_M = self.add_param(
            "BLOCK_M", IntegerBounds(min=32, max=config.M, step=4)
        )
        self.BLOCK_N = self.add_param("BLOCK_N", IntegerBounds(min=32, max=config.N))
        self.BLOCK_K2 = self.add_param("BLOCK_K2", IntegerBounds(min=32, max=config.K2))

        bytes_per_el = dtype_to_bytes(config.dtype)
        memory_constraint = (
            self.BLOCK_B * self.BLOCK_K2 * (self.BLOCK_N + 4) * bytes_per_el
            + self.BLOCK_B * self.BLOCK_K2 * (config.K1 + 4) * bytes_per_el
        ) - 65536
        self.add_constraint(memory_constraint, "memory_limit")

        # wg_x, wg_y, wg_z = (
        #     config.M // self.BLOCK_M,
        #     config.N // self.BLOCK_N,
        #     config.B // self.BLOCK_B,
        # )
        # num_workgroups = wg_x * wg_y * wg_z

    @override
    def load_wave_kernel(self):
        config = self.config

        if "f8" in config.dtype:
            base_attention, hyperparams, dynamic_symbols = (
                get_brevitas_pertensor_fp8_attention_kernel(
                    shape=bmnk1k2_to_attention_attributes(config),
                    f8_dtype=dtype_to_torch(config.dtype, self.target),
                    mfma_variant=self.mfma_variant.value,
                    dynamic_dims=False,
                )
            )
        else:
            base_attention, hyperparams, dynamic_symbols = get_vanilla_attention_kernel(
                shape=bmnk1k2_to_attention_attributes(config),
                mfma_variant=self.mfma_variant.value,
                dynamic_dims=False,
            )

        hyperparams.update(self._tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveKernel(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    @override
    def get_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            use_buffer_ops=True,
        )


class WaveAttentionGQABenchmark(WaveKernelBenchmark):
    config: AttentionConfigBSHD

    def setup_parameters(self):
        self.mfma_variant = self.add_param(
            "MFMA_VARIANT",
            CategoricalBounds(
                [
                    (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                    (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                    (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
                ]
            ),
            initial_value=0,
            include_hyperparam=False,
        )
        self.BLOCK_B = self.add_param("BLOCK_B", IntegerBounds(min=1, max=1, step=1))
        self.BLOCK_H = self.add_param("BLOCK_H", IntegerBounds(min=1, max=2, step=1))
        self.BLOCK_N_Q = self.add_param(
            "BLOCK_N_Q", IntegerBounds(min=16, max=128, step=16)
        )
        self.BLOCK_D_KV = self.add_param(
            "BLOCK_D_KV", IntegerBounds(min=16, max=128, step=16)
        )
        self.BLOCK_N_KV = self.add_param(
            "BLOCK_N_KV", IntegerBounds(min=16, max=64, step=16)
        )

    @override
    def load_wave_kernel(self):
        config = self.config

        base_attention, hyperparams, dynamic_symbols = get_gqa_bshd_attention_kernel(
            shape=bshd_to_attention_attributes(config),
            mfma_variant=self.mfma_variant.value,
            input_dtype=dtype_to_torch(config.dtype),
            output_dtype=dtype_to_torch("f32"),
        )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveKernel(
            launchable=base_attention,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    def get_compile_options(self):
        return WaveCompileOptions(
            schedule=SchedulingType.NONE,
            canonicalize=True,
            iree_launch_async=False,
        )


def create_extend_attention_inputs(
    shape: AttentionAttributes,
):

    dtype = torch.float16
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

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        output,
        b_req_idx,
        b_seq_len,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_offsets,
        b_start_loc,
        b_seq_len_prefix,
        extend_token_num,
        max_len_extend,
        logit_cap,
        rpe_bias,
        max_rpe_context_length,
    )


class WaveExtendAttentionBenchmark(WaveKernelBenchmark):
    config: AttentionConfigBMNK

    @override
    def setup_parameters(self):
        config = self.config

        mfma_bounds = CategoricalBounds(
            [
                (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
            ]
        )

        self.mfma_variant = self.add_param(
            "mfma_variant", mfma_bounds, initial_value=0, include_hyperparam=False
        )
        # self.add_param("BLOCK_H", IntegerBounds(min=1, max=config.B))
        # self.add_param("BLOCK_N_Q", IntegerBounds(min=32, max=config.M))
        # self.add_param("BLOCK_D_KV", IntegerBounds(min=32, max=config.N))
        # self.add_param("BLOCK_N_KV", IntegerBounds(min=32, max=config.K2))
        # self.add_param("BLOCK_S", IntegerBounds(min=32, max=config.K2))

        # bytes_per_el = dtype_to_bytes(config.dtype)
        # memory_constraint = (
        #     self.BLOCK_B * self.BLOCK_N * (self.BLOCK_K2 + 4) * bytes_per_el
        #     + self.BLOCK_B * self.BLOCK_K2 * (64 + 4) * bytes_per_el
        # ) - 65536
        # self.add_constraint(memory_constraint, "memory_limit")

    @override
    def load_wave_kernel(self):
        config = self.config
        shape = bmnk1k2_to_attention_attributes(config)
        dtype = dtype_to_torch(config.dtype)

        (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            output,
            b_req_idx,
            b_seq_len,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            b_start_loc,
            b_seq_len_prefix,
            extend_token_num,
            max_len_extend,
            logit_cap,
            _,
            _,
        ) = create_extend_attention_inputs(shape)
        shape = replace(shape, max_seq_len=max_len_extend)

        base_extend, hyperparams, dynamic_symbols = get_extend_attention_kernel(
            shape,
            self.mfma_variant.value,
            q_extend.shape,
            k_extend.shape,
            v_extend.shape,
            k_buffer.shape,
            v_buffer.shape,
            output.shape,
            logit_cap=logit_cap,
        )

        hyperparams.update(self.tuning_spec.hyperparams())
        hyperparams.update(get_default_scheduling_params())

        return WaveKernel(
            launchable=base_extend,
            hyperparams=hyperparams,
            dynamic_symbols=dynamic_symbols,
        )

    @override
    def get_compile_options(self):
        return WaveCompileOptions(
            canonicalize=True,
            schedule=SchedulingType.NONE,
            use_buffer_ops=True,
        )
