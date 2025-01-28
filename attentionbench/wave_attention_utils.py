from utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from attention_utils import AttentionConfig
import traceback

try:
    import iree.turbine.kernel as tk
    import iree.turbine.kernel.lang as tkl
    import iree.turbine.kernel.wave as tkw
    from iree.turbine.kernel.lang.global_symbols import *
    from iree.turbine.kernel.wave.constraints import MMAType
    from iree.turbine.kernel.wave.utils import (
        get_mfma_load_elems_per_thread,
        get_mfma_store_elems_per_thread,
    )
except ImportError:
    TURBINE_AVAILABLE = False
else:
    TURBINE_AVAILABLE = True

@dataclass
class AttentionShape:
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    head_size_kv: int
    # -----------------------
    # Prefill specific
    num_seqs: Optional[int] = None
    max_seq_len: Optional[int] = None
    total_seq_len: Optional[int] = None
    # -----------------------
    # Vanilla attention
    query_seq_len: Optional[int] = None
    kv_seq_len: Optional[int] = None

def get_vanilla_attention_kernel(
    shape: AttentionShape, mfma_variant: MMAType, dynamic_dims: bool, input_dtype: "dtype"
):
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
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD_QK = index_symbol("LOAD_ELEMS_PER_THREAD_QK")
    LOAD_ELEMS_PER_THREAD_PV = index_symbol("LOAD_ELEMS_PER_THREAD_PV")
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16 or mfma_variant[1] == MMAType.F32_16x16x32_F8:
        Mvec = 16
        Nvec = 16
    if mfma_variant[1] == MMAType.F32_32x32x8_F16 or mfma_variant[1] == MMAType.F32_32x32x16_F8:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
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

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, input_dtype],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, input_dtype],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, input_dtype],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, input_dtype)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD_PV)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD_QK: get_mfma_load_elems_per_thread(mfma_variant[0]),
        LOAD_ELEMS_PER_THREAD_PV: get_mfma_load_elems_per_thread(mfma_variant[1]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape.num_query_heads,
        M: shape.query_seq_len,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.kv_seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map


def compile_wave_attention_config(
    config: AttentionConfig, kernel_dir: Path, vmfb_dir: Path
) -> tuple[Path, Optional[Path]]:
    if not TURBINE_AVAILABLE:
        raise ValueError("iree.turbine package is not available")

    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")

    try:
        _compile_attention(config, mlir_file, vmfb_file)
    except Exception as e:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {config.get_name()}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        return mlir_file, None, None

    return mlir_file, vmfb_file


def _convert_dtype(dtype: str):
    dtypes = {
        "i8": tkl.i8,
        "i16": tkl.i16,
        "i32": tkl.i32,
        "i64": tkl.i64,
        "f8E4M3FNUZ": tkl.f8e4m3fnuz,
        "f16": tkl.f16,
        "f32": tkl.f32,
        "f64": tkl.f64,
        "bf16": tkl.bf16,
    }
    return dtypes[dtype]


def _compile_attention(config: AttentionConfig, mlir_file: Path, vmfb_file: Path):
    shape = AttentionShape(
        num_query_heads=config.B,
        num_kv_heads=config.B,
        query_seq_len=config.M,
        head_size_kv=config.N,
        head_size=config.K1,
        kv_seq_len=config.K2,
    )

    input_dtype = _convert_dtype(config.dtype)
    if input_dtype == tkl.f16:
        mfma_variant = (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    elif input_dtype == tkl.f8e4m3fnuz:
        mfma_variant = (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_F8)
    else:
        raise NotImplementedError(f"Got {config.dtype}, TK attention currently only support f8E4M3FNUZ and f16.")

    base_attention, hyperparams, _, _ = get_vanilla_attention_kernel(
        shape, mfma_variant, False, input_dtype
    )

    # config = get_default_run_config()
    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        run_config=config,
        schedule=False,
        inline=False,
    ):
        mod = base_attention().module_op  # This will generate vmfb file
        with open(mlir_file, "w") as f:
            f.write(str(mod))

        print(f"Successfully compiled to {vmfb_file}")
