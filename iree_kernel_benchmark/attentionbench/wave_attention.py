from ..utils import *
from .attention_config import AttentionAttributes, AttentionConfigBMNK
from .attention_utils import TuningSpec, redirect_stderr_to_file
from pathlib import Path
from typing import Optional
import math

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.scheduling.schedule_enums import SchedulingType


def get_custom_vanilla_attention_kernel(
    shape: AttentionAttributes,
    mfma_variant: MMAType,
    tuning_spec: TuningSpec,
    dynamic_dims: bool = False,
    is_causal: bool = False,
    is_v_transposed: bool = False,
    sliding_window_size: int = -1,
    scale: float = None,
):

    if sliding_window_size > 0 and not is_causal:
        raise NotImplementedError(
            "Sliding window is only supported for causal attention."
        )

    scale = scale or (1.0 / math.sqrt(shape.head_size))
    scale *= math.log2(math.e)

    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Workgroup tile sizes
    REDUCTION_B = tkl.sym.REDUCTION_B
    REDUCTION_M = tkl.sym.REDUCTION_M
    REDUCTION_N = tkl.sym.REDUCTION_N
    REDUCTION_K1 = tkl.sym.REDUCTION_K1
    REDUCTION_K2 = tkl.sym.REDUCTION_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    dims = [2, 0, 1, 3, 4]
    axes = [B, M, N, K1, K2]
    blocks = [BLOCK_B, BLOCK_M, BLOCK_N, BLOCK_K1, BLOCK_K2]
    reduction_tiles = [
        REDUCTION_B,
        REDUCTION_M,
        REDUCTION_N,
        REDUCTION_K1,
        REDUCTION_K2,
    ]

    M_waves = tuning_spec.M_warp
    N_waves = tuning_spec.N_warp

    constraints: list[tkw.Constraint] = []
    for wg_dim, wg_block_size in enumerate(tuning_spec.wg_tiles):
        if wg_block_size > 0:
            constraints.append(
                tkw.WorkgroupConstraint(axes[wg_dim], blocks[wg_dim], dims[wg_dim])
            )
    for wg_dim, reduction_block_size in enumerate(tuning_spec.reduction_tiles):
        if reduction_block_size > 0:
            constraints.append(
                tkw.TilingConstraint(axes[wg_dim], reduction_tiles[wg_dim])
            )
    if M_waves > 1:
        constraints.append(tkw.WaveConstraint(M, BLOCK_M // M_waves))
    if N_waves > 1:
        constraints.append(tkw.WaveConstraint(N, BLOCK_N // N_waves))

    # if mfma_variant[1] == MMAType.F32_16x16x16_F16:
    #     Mvec = 16
    #     Nvec = 16
    # if mfma_variant[1] == MMAType.F32_32x32x8_F16:
    Mvec = 32
    Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    # Value tensor mapping to transpose for efficient computation if the input is
    # not already transposed.
    v_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, K2: k}, outputs={B: i, N: j, K2: k}
    )

    def base_attention_core(q, k, v, c):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)
        sliding_window = tkl.Register[M, K2, tkl.i64](sliding_window_size)
        qk_scaling = tkl.Register[B, M, K2, tkl.f32](scale)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            k_reg = tkw.read(k)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            k2_index = tkw.self_index(K2, tkl.i64)
            mask = tkw.apply_expr(k2_index, lambda x: x < K2)
            mask = tkw.broadcast(mask, target_shape=[M, K2])
            if is_causal:
                # Indices i and j broadcasted along K2 with a twist:
                # here we use *static* information that is *implicitly* encoded
                # in the *transformation*: under the distribution constraints
                # specified we know that the shape [M] will eventually resolve
                # to [1] and can thus be "cast + broadcast" to [K2].
                m_index = tkw.self_index(M, tkl.i64)
                m_index = tkw.broadcast(m_index, target_shape=[M, K2])
                mask = (m_index >= k2_index) & mask
                if sliding_window_size > 0:
                    mask = (m_index - k2_index <= sliding_window) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            x_j *= qk_scaling
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            if is_v_transposed:
                v_reg = tkw.read(v)
            else:
                v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    @tkw.wave(constraints)
    def base_attention_transposed_v(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    shape_bmnk = shape.to_bmnk1k2()

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: tuning_spec.wg_tiles[0],
        BLOCK_M: tuning_spec.wg_tiles[1],
        BLOCK_N: tuning_spec.wg_tiles[2],
        BLOCK_K1: tuning_spec.wg_tiles[3],
        BLOCK_K2: tuning_spec.wg_tiles[4],
        REDUCTION_B: tuning_spec.reduction_tiles[0],
        REDUCTION_M: tuning_spec.reduction_tiles[1],
        REDUCTION_N: tuning_spec.reduction_tiles[2],
        REDUCTION_K1: tuning_spec.reduction_tiles[3],
        REDUCTION_K2: tuning_spec.reduction_tiles[4],
        B: shape_bmnk.B,
        M: shape_bmnk.M,
        N: shape_bmnk.N,
        K1: shape_bmnk.K1,
        K2: shape_bmnk.K2,
    }

    dynamic_symbols = []
    if dynamic_dims:
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    if is_v_transposed:
        base_attention = base_attention_transposed_v

    return base_attention, hyperparams, dynamic_symbols


def compile_attention_wave_vanilla(
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

    base_attention, hyperparams, dynamic_symbols = get_custom_vanilla_attention_kernel(
        shape=shape, mfma_variant=mfma_variant, tuning_spec=spec, dynamic_dims=False
    )

    hyperparams.update(get_default_scheduling_params())

    config = shape.to_bmnk1k2()

    compile_options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        backend="rocm",
        target="gfx942",
        schedule=SchedulingType.NONE,
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
