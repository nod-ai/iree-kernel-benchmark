from kernel_bench.core.template import KernelBenchmark

from typing import Optional
import torch
import triton
import time

from aiter.ops.triton.prefill_attention import context_attention_fwd
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.extend_attention import extend_attention_fwd
from kernel_bench.kernels.attention.attention_config import (
    AttentionConfigExtend,
    ExtendAttentionInputs,
)
from kernel_bench.utils.torch_utils import benchmark_function_torch

# def extend_attention_fwd(
#     q_extend: torch.Tensor,
#     k_extend: torch.Tensor,
#     v_extend: torch.Tensor,
#     o_extend: torch.Tensor,
#     k_buffer: torch.Tensor,
#     v_buffer: torch.Tensor,
#     qo_indptr: torch.Tensor,
#     kv_indptr: torch.Tensor,
#     kv_indices: torch.Tensor,
#     max_len_extend: int,
#     custom_mask: torch.Tensor = None,
#     is_causal=False,
#     mask_indptr: torch.Tensor = None,
#     sm_scale=None,
#     logit_cap=0.0,
#     skip_prefix_custom_mask=True,
#     config: Optional[dict[str, any]] = None,
#     setup_only: bool = False,
# ) -> tuple:
#     """
#     q_extend, k_extend, v_extend, o_extend: contiguous tensors

#     k_buffer, v_buffer: (prefix + extend) tensors in mem_manager

#     If setup_only=True, returns (grid, kernel_args, kernel_kwargs) for later execution.
#     If setup_only=False, executes the kernel and returns None.
#     """

#     Lq, Lv = (
#         q_extend.shape[-1],
#         v_extend.shape[-1],
#     )

#     if Lq == 576:
#         BLOCK_DMODEL = 512
#         BLOCK_DPE = 64
#     elif Lq == 288:
#         BLOCK_DMODEL = 256
#         BLOCK_DPE = 32
#     elif Lq == 192:
#         BLOCK_DMODEL = 128
#         BLOCK_DPE = 64
#     else:
#         BLOCK_DMODEL = triton.next_power_of_2(Lq)
#         BLOCK_DPE = 0
#     BLOCK_DV = triton.next_power_of_2(Lv) // 2

#     sm_scale = sm_scale or 1.0 / (Lq**0.5)
#     batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
#     kv_group_num = q_extend.shape[1] // k_extend.shape[1]

#     USE_CUSTOM_MASK = custom_mask is not None
#     # Skip custom mask for prefix part
#     SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

#     if config is None:
#         config = _get_config(HEAD_SIZE=Lq, dtype=q_extend.dtype)

#     config["BLOCK_M"] = 32

#     num_blocks = triton.cdiv(max_len_extend, config["BLOCK_M"])
#     grid = (head_num * num_blocks * batch_size,)

#     kernel_args = (
#         q_extend,
#         k_extend,
#         v_extend,
#         o_extend,
#         k_buffer,
#         v_buffer,
#         qo_indptr,
#         kv_indptr,
#         kv_indices,
#         custom_mask,
#         mask_indptr,
#         sm_scale,
#         kv_group_num,
#         q_extend.stride(0),
#         q_extend.stride(1),
#         k_extend.stride(0),
#         k_extend.stride(1),
#         v_extend.stride(0),
#         v_extend.stride(1),
#         o_extend.stride(0),
#         o_extend.stride(1),
#         k_buffer.stride(0),
#         k_buffer.stride(1),
#         v_buffer.stride(0),
#         v_buffer.stride(1),
#     )

#     kernel_kwargs = {
#         "logit_cap": logit_cap,
#         "BLOCK_DMODEL": BLOCK_DMODEL,
#         "BLOCK_DPE": BLOCK_DPE,
#         "BLOCK_DV": BLOCK_DV,
#         "Lq": Lq,
#         "Lv": Lv,
#         "USE_CUSTOM_MASK": USE_CUSTOM_MASK,
#         "IS_CAUSAL": is_causal,
#         "SKIP_PREFIX_CUSTOM_MASK": SKIP_PREFIX_CUSTOM_MASK,
#         "STORE_TRANSPOSE": True,
#         "NUM_Q_HEADS": head_num,
#         "NUM_BLOCKS": num_blocks,
#         "BATCH": batch_size,
#         "NUM_XCDS": get_num_xcds(),
#         **config,
#     }

#     if setup_only:
#         return grid, kernel_args, kernel_kwargs
#     else:
#         _fwd_kernel[grid](*kernel_args, **kernel_kwargs)
#         return None


# def triton_extend_attention(inputs: ExtendAttentionInputs):
#     extend_attention_fwd(
#         q_extend=inputs.q_extend,
#         k_extend=inputs.k_extend,
#         v_extend=inputs.v_extend,
#         o_extend=inputs.output,
#         k_buffer=inputs.k_buffer,
#         v_buffer=inputs.v_buffer,
#         qo_indptr=inputs.qo_indptr,
#         kv_indptr=inputs.kv_indptr,
#         kv_indices=inputs.kv_indices,
#         custom_mask=None,
#         is_causal=None,
#         mask_indptr=None,
#         max_len_extend=inputs.max_len_extend,
#         logit_cap=inputs.logit_cap,
#     )


class TritonExtendAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigExtend

    def run_bench(self, device, num_iterations=1, timeout=None):
        inputs = self.config.get_inputs()

        try:
            mean_time_us = benchmark_function_torch(
                extend_attention_fwd,
                iterations=num_iterations,
                # Extend attention inputs
                q_extend=inputs.q_extend,
                k_extend=inputs.k_extend,
                v_extend=inputs.v_extend,
                o_extend=inputs.output,
                k_buffer=inputs.k_buffer,
                v_buffer=inputs.v_buffer,
                qo_indptr=inputs.qo_indptr,
                kv_indptr=inputs.kv_indptr,
                kv_indices=inputs.kv_indices,
                custom_mask=None,
                is_causal=None,
                mask_indptr=None,
                max_len_extend=inputs.max_len_extend,
                logit_cap=inputs.logit_cap,
                config={
                    "BLOCK_M": 16,
                    "BLOCK_N": 16,
                    "waves_per_eu": 0,
                    "num_warps": 4,
                    "matrix_instr_nonkdim": 16,
                    "kpack": 2,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
