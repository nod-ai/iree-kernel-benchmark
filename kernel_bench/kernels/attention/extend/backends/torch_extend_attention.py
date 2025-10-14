from enum import Enum
import math
import torch
from torch.nn import functional as F
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.config.types.attention import AttentionConfigExtend
from kernel_bench.utils.torch_utils import benchmark_function_torch
from wave_lang.kernel.wave.utils.torch_utils import device_empty, device_zeros


class TorchExtendAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigExtend

    def run_bench(self, device, num_iterations, timeout=None):
        inputs = self.config.get_inputs(self.device_ctx)
        dtype = self.device_ctx.dtype_to_torch(self.config.dtype)

        try:
            mean_time_us = benchmark_function_torch(
                torch_extend_attn,
                iterations=50,
                compile=False,
                # Extend attention inputs
                q_extend=inputs.q_extend,
                k_buffer=inputs.k_buffer,
                v_buffer=inputs.v_buffer,
                b_req_idx=inputs.b_req_idx,
                b_start_loc=inputs.b_start_loc,
                b_seq_len=inputs.b_seq_len,
                b_seq_len_prefix=inputs.b_seq_len_prefix,
                max_len_extend=inputs.max_len_extend,
                extend_token_num=inputs.extend_token_num,
                dtype=dtype,
                is_causal=False,
                logit_cap=inputs.logit_cap,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)


class ScoreMod(Enum):
    SoftCap = 0
    RPE = 1


def torch_extend_attn(
    q_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_seq_len_prefix: torch.Tensor,
    max_len_extend: int,
    extend_token_num: int,
    dtype: torch.dtype,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    rpe_bias: torch.Tensor = None,
    score_mod: ScoreMod = ScoreMod.SoftCap,
    max_rpe_context_length: int = 0,
) -> torch.Tensor:
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = device_empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )
    o_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer,
        k_buffer,
        v_buffer,
        o_buffer,
        b_start_loc,
        b_seq_len,
        max_len_extend,
        is_causal,
        logit_cap=logit_cap,
        rpe_bias=rpe_bias,
        score_mod=score_mod,
        max_rpe_context_length=max_rpe_context_length,
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend

    return o_extend


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_extend: int,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    rpe_bias: torch.Tensor = None,
    score_mod: ScoreMod = ScoreMod.SoftCap,
    max_rpe_context_length: int = 0,
):

    cu_seq_lens = [0] * (len(b_seq_len) + 1)
    for i, seq_len in enumerate(b_seq_len):
        cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    for i in range(len(b_seq_len)):
        start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
        qkv_len = end - start
        Q = q[start:end].permute(1, 0, 2)
        K = k[start:end].permute(1, 0, 2)
        K = K.repeat_interleave(Q.shape[0] // K.shape[0], dim=0)
        V = v[start:end].permute(1, 0, 2)
        V = V.repeat_interleave(Q.shape[0] // V.shape[0], dim=0)
        dk_sqrt = math.sqrt(1.0 / Q.shape[-1])
        a = torch.bmm(Q * dk_sqrt, K.transpose(-1, -2))
        if score_mod == ScoreMod.SoftCap:
            a = a / logit_cap
            a = torch.tanh(a)
            a = a * logit_cap
        else:
            rpe_cond = t5_rpe_masked_cond(
                rpe_bias,
                max_rpe_context_length=max_rpe_context_length,
                sequence_length=K.shape[1],
            )
            rpe_cond = rpe_cond.unsqueeze(0)
            rpe_cond = rpe_cond.expand(Q.shape[0], *rpe_cond.shape[1:])
            a = a + rpe_cond
        if is_causal:
            # Create a mask for the upper triangular part (excluding the diagonal)
            mask = (
                torch.triu(torch.ones(a.shape[-2:]), diagonal=1)
                .unsqueeze(0)
                .expand(a.shape)
            )
            # Apply the mask to set the upper triangular part to -infinity
            a[mask == 1] = float("-inf")
        reference = torch.bmm(F.softmax(a, dim=-1).to(dtype=V.dtype), V)
        reference = reference.squeeze(0).permute(1, 0, 2)
        o[start:end] = reference

    return o


def t5_rpe_masked_cond(
    rpe: torch.Tensor, max_rpe_context_length: int, sequence_length: int
) -> torch.Tensor:
    positions = torch.arange(sequence_length).to(device=rpe.device)
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = ((pos_diff >= 0) & (pos_diff < max_rpe_context_length)).to(device=rpe.device)
    rpe_cond = device_zeros(sequence_length, sequence_length, dtype=rpe.dtype)
    rpe_cond[mask] = rpe[pos_diff[mask]]
    return rpe_cond
