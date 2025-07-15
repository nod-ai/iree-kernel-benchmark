from ..utils import *
from .attention_config import AttentionConfigBSHD, AttentionAttributes
from .attention_utils import TuningSpec, redirect_stderr_to_file

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum
import math
import torch

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.scheduling.schedule_enums import SchedulingType
from typing import Optional

DTYPE_TO_WAVE = {
    "bf16": tkl.bf16,
    "f8e5m2": tkl.f8e5m2,
    "f8e5m2fnuz": tkl.f8e5m2fnuz,
    "f8e4m3fn": tkl.f8e4m3fn,
    "f8e4m3fnuz": tkl.f8e4m3fnuz,
    "f16": tkl.f16,
    "f32": tkl.f32,
    "f64": tkl.f64,
    "i16": tkl.i16,
    "i32": tkl.i32,
    "i64": tkl.i64,
    "bool": tkl.bool,
}


def dtype_to_wave(dtype: str) -> tkw.DataType:
    try:
        return DTYPE_TO_WAVE[dtype]
    except KeyError:
        raise ValueError(f"Unable to map {dtype} to Wave datatype.")


def get_gqa_bshd_attention_kernel(
    shape: AttentionAttributes,
    mfma_variant: tuple[MMAType, MMAType],
    tuning_spec: TuningSpec,
    is_causal: Optional[bool] = False,
    layer_scaling: Optional[float] = None,
    sliding_window_size: Optional[int] = -1,
    q_scale: Optional[float] = 1.0,
    k_scale: Optional[float] = 1.0,
    v_scale: Optional[float] = 1.0,
    use_fp8: Optional[bool] = False,
):

    if sliding_window_size > 0 and not is_causal:
        raise NotImplementedError(
            "Sliding window is only supported for causal attention."
        )

    input_dtype = shape.dtype
    output_dtype = "f32"

    # Determine dtype of operands.
    wave_input_dtype = dtype_to_wave(input_dtype)
    wave_output_dtype = dtype_to_wave(output_dtype)

    LOG2E = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    layer_scaling = (layer_scaling or dk_sqrt) * LOG2E

    f8_dtype: torch.dtype = torch.float8_e4m3fnuz
    F8_DTYPE = dtype_to_wave("f8e4m3fnuz")
    F8_MAX = torch.finfo(f8_dtype).max

    # maximum expected value from attention softmax
    ATTENTION_SOFTMAX_MAX = 1.0

    # FP8 offset
    # If we need to truncate to fp8 post softmax we apply a scaling to use the
    # full fp8 range. We can do this with a offset as post `exp2` this equates
    # to multiplying by a static value. We are able to do this as `max` and
    # `sum` are scaled by the same value so the end result is the same.
    FP8_OFFSET_VAL = math.log2(F8_MAX / ATTENTION_SOFTMAX_MAX)

    # Dequant Tensor Scaling
    DEQUANT_QK = q_scale * k_scale
    if use_fp8:
        layer_scaling *= DEQUANT_QK

    # Clamp input to dstTy(usually `fp8`) MAX value to prevent NaNs.
    # We do not clamp for `-MAX` because this function meant to only be
    # used by attention's exp2 who's value is always > 0.
    def low_precision_clamp(source_reg, upper_bound):
        clamped = tkw.minimum(source_reg, upper_bound)
        return tkw.cast(clamped, F8_DTYPE)

    B = tkl.sym.B
    H = tkl.sym.H  # number of heads
    H_Q = tkl.sym.H_Q  # number of query heads
    H_KV = tkl.sym.H_KV  # number of key/value heads
    N_Q = tkl.sym.N_D  # query sequence length
    N_KV = tkl.sym.N_KV  # key/value sequence length
    D_Q = tkl.sym.D_Q  # query head size
    D_KV = tkl.sym.D_KV  # key/value head size

    # And their corresponding tile sizes.
    BLOCK_H = tkl.sym.BLOCK_H
    BLOCK_H_Q = tkl.sym.BLOCK_H_Q
    BLOCK_H_KV = tkl.sym.BLOCK_H_KV
    BLOCK_N_Q = tkl.sym.BLOCK_N_Q
    BLOCK_N_KV = tkl.sym.BLOCK_N_KV
    BLOCK_D_Q = tkl.sym.BLOCK_D_Q
    BLOCK_D_KV = tkl.sym.BLOCK_D_KV
    BLOCK_B = tkl.sym.BLOCK_B
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(N_Q, BLOCK_N_Q, 0)]
    constraints += [tkw.WorkgroupConstraint(D_KV, BLOCK_D_KV, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.WorkgroupConstraint(H_KV, BLOCK_H, 3, primary=False)]
    if is_causal:
        constraints += [
            tkw.TilingConstraint(
                N_KV,
                BLOCK_N_KV,
                iters=sympy.Min((WORKGROUP_0 + 1) * BLOCK_N_Q, N_KV) // BLOCK_N_KV,
            )
        ]
    else:
        constraints += [
            tkw.TilingConstraint(
                N_KV,
                BLOCK_N_KV,
            )
        ]
    constraints += [tkw.WaveConstraint(N_Q, BLOCK_N_Q / 4)]
    constraints += [tkw.WaveConstraint(D_KV, BLOCK_D_KV / 1)]

    if "16x16" in mfma_variant[1].name or "16x16" in mfma_variant[0].name:
        Mvec = 16
        Nvec = 16
    else:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, H: 0, H_KV: 0, N_Q: Mvec, D_KV: Nvec},
        )
    ]

    head_ratio = shape.num_query_heads // shape.num_kv_heads
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, D_KV: k, N_Q: l},
        outputs={B: i, N_Q: l, H: j, D_KV: k},
    )
    q_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N_Q: k, D_Q: l},
        outputs={B: i, H: j, N_Q: k, D_Q: l},
    )
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, N_KV: k, D_Q: l},
        outputs={B: i, H_KV: j, N_KV: k, D_Q: l},
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, D_KV: k, N_KV: l},
        outputs={B: i, H_KV: j, D_KV: k, N_KV: l},
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype],
        k: tkl.Memory[B, N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype],
        v: tkl.Memory[B, N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype],
        c: tkl.Memory[B, N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, wave_output_dtype],
    ):

        qk_scaling = tkl.Register[B, H, N_Q, N_KV, tkl.f32](layer_scaling)
        v_dequant = tkl.Register[B, D_KV, N_Q, tkl.f32](v_scale)
        fp8_offset = tkl.Register[B, N_Q, N_KV, tkl.f32](FP8_OFFSET_VAL)
        fp8_max = tkl.Register[B, N_Q, N_KV, tkl.f32](F8_MAX)
        c_reg = tkl.Register[B, H, D_KV, N_Q, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, N_Q, tkl.f32](0.0)
        init_max = tkl.Register[B, H, N_Q, tkl.f32](-1e6)
        sliding_window = tkl.Register[N_Q, N_KV, tkl.i32](sliding_window_size)
        ZEROF = tkl.Register[N_Q, N_KV, tkl.f32](0.0)
        MIN_INF = tkl.Register[N_Q, N_KV, tkl.f32](-1e6)

        @tkw.iterate(N_KV, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, N_Q, tkl.f32],
            partial_sum: tkl.Register[B, H, N_Q, tkl.f32],
            acc: tkl.Register[B, H, D_KV, N_Q, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, N_KV, N_Q, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            k_reg = tkw.read(k, mapping=k_mapping)
            if use_fp8:
                q_reg = tkw.cast(q_reg, F8_DTYPE)
                k_reg = tkw.cast(k_reg, F8_DTYPE)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, N_Q, N_KV])
            k2_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < N_KV)
            mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
            if is_causal:
                m_index = tkw.self_index(N_Q, tkl.i32)
                m_index = tkw.broadcast(m_index, target_shape=[N_Q, N_KV])
                mask = (m_index >= k2_index) & mask
                if sliding_window_size > 0:
                    mask = (m_index - k2_index <= sliding_window) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            x_j *= qk_scaling
            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            if use_fp8:
                e_delta = tkw.exp2(x_j - m_j + fp8_offset)
            else:
                e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            if use_fp8:
                imm_f = low_precision_clamp(e_delta, fp8_max)
            else:
                imm_f = tkw.cast(e_delta, wave_input_dtype)
            v_reg = tkw.read(v, mapping=v_mapping)
            if use_fp8:
                v_reg = tkw.cast(v_reg, F8_DTYPE)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        if use_fp8:
            res = res_mm * reciprocal_sum * v_dequant
        else:
            res = res_mm * reciprocal_sum
        if wave_output_dtype != tkl.f32:
            res = tkw.cast(res, wave_output_dtype)
        tkw.write(res, c, mapping=mapping)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: tuning_spec.wg_tiles[0],  # 1,
        BLOCK_H: tuning_spec.wg_tiles[1],  # 1,
        BLOCK_N_Q: tuning_spec.wg_tiles[2],  # 128,
        BLOCK_D_KV: tuning_spec.wg_tiles[3],  # 128,
        BLOCK_N_KV: tuning_spec.wg_tiles[4],  # 32,
        B: shape.num_seqs,
        H: shape.num_query_heads,
        H_KV: shape.num_kv_heads,
        N_Q: shape.query_seq_len,
        D_KV: shape.head_size_kv,
        D_Q: shape.head_size,
        N_KV: shape.kv_seq_len,
    }

    dynamic_symbols = []

    return base_attention, hyperparams, dynamic_symbols


def compile_attention_wave_bshd(
    shape: AttentionAttributes,
    spec: TuningSpec,
    mlir_file: Path,
    vmfb_file: Path,
    dump_dir: Optional[Path],
    mfma_variant: tuple[MMAType] = (
        MMAType.F32_32x32x16_K8_F16,
        MMAType.F32_32x32x8_F16,
    ),
) -> tuple[Path, Optional[Path]]:

    base_attention, hyperparams, dynamic_symbols = get_gqa_bshd_attention_kernel(
        shape=shape, mfma_variant=mfma_variant, tuning_spec=spec
    )

    hyperparams.update(get_default_scheduling_params())

    config = shape.to_bshd()

    compile_options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        backend="rocm",
        target="gfx942",
        print_ir_after_all=dump_dir is not None,
    )

    if dump_dir:
        dump_file = dump_dir / "wave" / (config.get_name() + ".debug.mlir")
        with redirect_stderr_to_file(dump_file):
            result = wave_compile(compile_options, base_attention)
    else:
        result = wave_compile(compile_options, base_attention)

    with open(mlir_file, "w") as mlir_out:
        mlir_out.write(result.asm)

    return mlir_file, vmfb_file
