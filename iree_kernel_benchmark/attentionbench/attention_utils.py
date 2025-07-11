from ..utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum
import math

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.templates.vanilla_attention import get_vanilla_attention_kernel
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.scheduling.schedule_enums import SchedulingType
from typing import Optional

class IntrinsicType(Enum):
    """
    Formatting for different target intrinsics:
        <kind>_<elem-type-C>_<M>x<N>x<K>_<elem-type-A>[_<elem-type-B>]

    Values: 0xABCD where:
    * A = vendor:
    * 1 = AMD
    * 2 = NVIDIA
    * B = architecture. When an intrinsic exists in multiple architectures, this
        should be the architecture it was introduced in, as long as it still
        has the same semantics. If a new architecture breaks an existing
        intrinsic's semantics, we can use that field for versioning.
    * For AMD:
        * 0 = CDNA1
        * 1 = CDNA2
        * 2 = CDNA3
        * 8 = RDNA3
    * C = element type of A-matrix:
    * 0 = 64-bit float (e.g. IEEE754 double precision)
    * 1 = 32-bit float (e.g. IEEE754 single precision, and "xf32" fast variants)
    * 2 = 16-bit float (incl. IREE754 half and bf16)
    * 3 = 8-bit float (incl. f8E5M2, f8E4M3, and "FNUZ" variants)
    * C = 8-bit integer (any signedness)
    * D enumerates intrinsics that share the same 0xABC* bits.
    """

    # Intrinsics introduced in CDNA1
    MFMA_F32_16x16x16_F16 = 0x1020
    MFMA_F32_32x32x8_F16 = 0x1021
    VMFMA_F32_32x32x16_F16 = 0x1022
    MFMA_I32_16x16x16_I8 = 0x10C0
    MFMA_I32_32x32x8_I8 = 0x10C1

    # Intrinsics introduced in CDNA3
    MFMA_F32_16x16x32_F8 = 0x1230
    MFMA_F32_32x32x16_F8 = 0x1231
    MFMA_I32_16x16x32_I8 = 0x12C0
    MFMA_I32_32x32x16_I8 = 0x12C1


def get_intrinsic_string(intrinsic: IntrinsicType):
    match intrinsic:
        case IntrinsicType.VMFMA_F32_32x32x16_F16:
            return f"#iree_gpu.virtual_mma_layout<intrinsic = {intrinsic.name}>"
        case _:
            return f"#iree_gpu.mma_layout<{intrinsic.name}>"


def get_pv_intrinsic(intrinsic: IntrinsicType):
    """
    QK intrinsics and PV intrinsics can differ. Mostly used for
    selecting VMFMA for QK to maximize contiguous read from shared memory.
    """
    match intrinsic:
        case IntrinsicType.VMFMA_F32_32x32x16_F16:
            return IntrinsicType.MFMA_F32_32x32x8_F16
        case _:
            return intrinsic


@dataclass
class AttentionConfig:
    B: int
    M: int
    N: int
    K1: int
    K2: int
    dtype: str

    def get_name(self) -> str:
        return f"attention_{self.B}x{self.M}x{self.N}x{self.K1}x{self.K2}x{self.dtype}"

    def get_query_shape(self) -> str:
        return f"{self.B}x{self.M}x{self.K1}x{self.dtype}"

    def get_key_shape(self) -> str:
        return f"{self.B}x{self.K2}x{self.K1}x{self.dtype}"

    def get_value_shape(self) -> str:
        return f"{self.B}x{self.K2}x{self.N}x{self.dtype}"

    def get_output_shape(self) -> str:
        return f"{self.B}x{self.M}x{self.N}x{self.dtype}"

    def get_byte_count(self) -> int:
        dtype_bits_map = {
            "f32": 32,
            "f16": 16,
            "bf16": 16,
            "f8E4M3FNUZ": 8,
            "i8": 8,
            "i32": 32,
        }
        bytes_per_element = dtype_bits_map[self.dtype] // 8
        element_count = (
            (self.B * self.M * self.K1)
            + (self.B * self.K2 * self.K1)
            + (self.B * self.K2 * self.N)
            + (self.B * self.M * self.N)
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        # We measure flops of the two matmuls only
        qk_matmul_flops = 2 * self.B * self.M * self.K2 * self.K1
        pv_matmul_flops = 2 * self.B * self.M * self.N * self.K2
        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops


@dataclass
class TuningSpec:
    wg_tiles: list[int]
    reduction_tiles: list[int]
    M_warp: int
    N_warp: int
    intrinsic: str
    waves_per_eu: Optional[int]
    denorm_flush: bool

    def get_lowering_config(self) -> str:
        return (
            f"#iree_gpu.lowering_config<"
            + "{ "
            + f"workgroup = [{', '.join(map(str, self.wg_tiles))}], "
            + f"reduction = [{', '.join(map(str, self.reduction_tiles))}],"
            + f"promote_operands = [1, 2]"
            + " }"
            + f">"
        )

    def get_translation_info(self) -> str:
        llvm_func_attrs = []
        if self.waves_per_eu:
            llvm_func_attrs += [f'"amdgpu-waves-per-eu" = "{self.waves_per_eu}"']
        if self.denorm_flush:
            llvm_func_attrs += [f'"denormal-fp-math-f32" = "preserve-sign"']
        return (
            f"#iree_codegen.translation_info<"
            + f"pipeline = LLVMGPUVectorDistribute"
            + f" workgroup_size = [{self.N_warp * self.M_warp * 64}]"
            + f" subgroup_size = 64"
            + f" , {{llvm_func_attrs = {{ {','.join(llvm_func_attrs)} }}"
            + f"}}"
            + f">"
        )

    def get_compilation_info(self) -> str:
        return (
            f"#iree_codegen.compilation_info<"
            + f"lowering_config = {self.get_lowering_config()}"
            + f", translation_info = {self.get_translation_info()}"
            + f">"
        )

    def get_qk_config_info(self) -> str:
        return (
            f"#iree_gpu.lowering_config<{{"
            + f"mma_kind = {get_intrinsic_string(self.intrinsic)}"
            + f", subgroup_m_count = {self.M_warp}"
            + f", subgroup_n_count = {self.N_warp}"
            + f", promote_operands = [1]"
            + f"}}>"
        )

    def get_pv_config_info(self) -> str:
        return (
            f"#iree_gpu.lowering_config<{{"
            + f"mma_kind = {get_intrinsic_string(get_pv_intrinsic(self.intrinsic))}"
            + f", subgroup_m_count = {self.M_warp}"
            + f", subgroup_n_count = {self.N_warp}"
            + f", promote_operands = [1]"
            + f"}}>"
        )


def generate_mlir(config: AttentionConfig, tuning: Optional[TuningSpec] = None):
    shapes = f"""\
!dtype = {config.dtype}
!Q     = tensor<{config.get_query_shape()}>
!K     = tensor<{config.get_key_shape()}>
!V     = tensor<{config.get_value_shape()}>
!O     = tensor<{config.get_output_shape()}>
"""

    spec = ""
    if tuning and config.dtype == "f16":
        spec = f"""\
#tuning = {tuning.get_compilation_info()}
"""

    attn_kernel = f"""
#Q = affine_map<(b, m, n, k1, k2) -> (b, m, k1)>
#K = affine_map<(b, m, n, k1, k2) -> (b, k2, k1)>
#V = affine_map<(b, m, n, k1, k2) -> (b, k2, n)>
#S = affine_map<(b, m, n, k1, k2) -> ()>
#O = affine_map<(b, m, n, k1, k2) -> (b, m, n)>

func.func @main(%Q : !Q, %K : !K, %V : !V) -> !O {{
  %scale = arith.constant 1.0 : !dtype
  %empty = tensor.empty() : !O
  %O = iree_linalg_ext.attention
       {{ indexing_maps = [#Q, #K, #V, #S, #O]
          ,decomposition_config = {{
           qk_attrs = {{attention_qk_matmul, lowering_config = {tuning.get_qk_config_info()}}},
           pv_attrs = {{attention_pv_matmul, lowering_config = {tuning.get_pv_config_info()}}}
         }}
         {",compilation_info = #tuning" if tuning and config.dtype == "f16" else ""}
       }}
       ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype) outs(%empty : !O) {{
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
        }} -> !O
  return %O : !O
}}
"""
    mlir_template = shapes + "\n" + spec + "\n" + attn_kernel
    return mlir_template


def get_attention_flags() -> list[str]:
    # return ["--iree-codegen-gpu-native-math-precision"]
    return []

def get_custom_vanilla_attention_kernel(
    shape: AttentionShape,
    mfma_variant: MMAType,
    tuning_spec: TuningSpec,
    dynamic_dims: bool,
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

    axes = [B, M, N, K1, K2]
    blocks = [BLOCK_B, BLOCK_M, BLOCK_N, BLOCK_K1, BLOCK_K2]
    reduction_tiles = [REDUCTION_B, REDUCTION_M, REDUCTION_N, REDUCTION_K1, REDUCTION_K2]

    M_waves = tuning_spec.M_warp
    N_waves = tuning_spec.N_warp

    constraints: list[tkw.Constraint] = []
    for wg_dim, wg_block_size in enumerate(tuning_spec.wg_tiles):
        if wg_block_size > 0:
            constraints.append(
                tkw.WorkgroupConstraint(axes[wg_dim], blocks[wg_dim], wg_dim))
        # else:
        #     constraints.append(
        #         tkw.WorkgroupConstraint(axes[wg_dim], axes[wg_dim], wg_dim))
    for wg_dim, reduction_block_size in enumerate(tuning_spec.reduction_tiles):
        if reduction_block_size > 0:
            constraints.append(tkw.TilingConstraint(axes[wg_dim], reduction_tiles[wg_dim]))
    if M_waves > 1:
        constraints.append(tkw.WaveConstraint(M, BLOCK_M / M_waves))
    if N_waves > 1:
        constraints.append(tkw.WaveConstraint(N, BLOCK_N / N_waves))

    # constraints.append(tkw.WorkgroupConstraint(M, BLOCK_M, 0))
    # constraints.append(tkw.WorkgroupConstraint(N, BLOCK_N, 1))
    # constraints.append(tkw.WorkgroupConstraint(B, BLOCK_B, 2))
    # constraints.append(tkw.TilingConstraint(K2, BLOCK_K2))
    # constraints.append(tkw.WaveConstraint(M, BLOCK_M / 4))
    # constraints.append(tkw.WaveConstraint(N, BLOCK_N / 1))

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
        B: shape.num_query_heads,
        M: shape.query_seq_len,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.kv_seq_len,
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

def generate_tk_attention_mlir(
        config: AttentionConfig, 
        spec: TuningSpec,
        vmfb_file: Path,
        dump_dir: Path,
        mfma_variant: tuple[MMAType] = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16)):
    attention_shape = AttentionShape(
        num_query_heads=config.B,
        num_kv_heads=config.B,
        head_size=config.K1,
        head_size_kv=config.N,
        batch_size=None,
        query_seq_len=config.M,
        kv_seq_len=config.K2,
        num_seqs=None,
        max_seq_len=None,
        total_seq_len=None,
        context_len=None,
        fixed_seq_len_prefix=None,
        fixed_seq_len_extend=None,
        block_size=None
    )

    base_attention, hyperparams, dynamic_symbols = \
        get_custom_vanilla_attention_kernel(shape=attention_shape, 
                                            mfma_variant=mfma_variant,
                                            tuning_spec=spec,
                                            dynamic_dims=False)

    hyperparams.update(get_default_scheduling_params())

    if dump_dir:
        phase_dump = dump_dir / 'wave' / config.get_name()

    compile_options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        backend="rocm",
        target="gfx942",
        schedule=SchedulingType.NONE,
        print_ir_after_all=False,
        dump_phases=phase_dump if dump_dir else None
    )
    result = wave_compile(compile_options, base_attention)

    return result.asm

def compile_attention_config(
    config: AttentionConfig, 
    kernel_dir: Path, 
    vmfb_dir: Path, 
    dump_dir: Path = None,
    extra_compiler_args: list[str] = [],
    tk: bool = False,
    mfma_variant: tuple[MMAType] = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
) -> tuple[Path, Optional[Path]]:
    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")
    dump_file = kernel_dir / (config.get_name() + ".stderr.mlir")

    # TODO: Use different tuning specs for different configs. This is just a
    # general tuning config that worked well for sdxl shapes.

    num_workgroups = 8
    waves_per_workgroup = 4
    tiles_per_workgroup = 32

    spec = TuningSpec(
        [1, 256, 0, 0, 0],
        [0, 0, 0, 0, 32],
        waves_per_workgroup,
        1,
        IntrinsicType.VMFMA_F32_32x32x16_F16,
        2,
        True,
    )
    # Generate mlir content
    if tk:
        mlir_content = generate_tk_attention_mlir(
            config, spec, vmfb_file, dump_dir, mfma_variant)
    else:
        mlir_content = generate_mlir(config, spec)

    # Write MLIR content to file
    with open(mlir_file, "w") as f:
        f.write(mlir_content)

    if tk:
        return mlir_file, vmfb_file

    # TODO: Do not hardcode device information, instead pass it as a class
    # Compile MLIR to vmfb
    exec_args = [
        "iree-compile",
        # Input file
        f"{mlir_file}",
        # Output file
        "-o",
        f"{vmfb_file}",
        # Target Device: hip
        "--iree-hal-target-device=hip",
        # Device: MI300x
        "--iree-hip-target=gfx942",
    ] + get_attention_flags()
    if dump_dir:
        phase_dump = dump_dir / 'iree' / config.get_name()
        exec_args.append(f'--dump-compilation-phases-to={phase_dump}')

    print(" ".join(exec_args))

    ret_value, stdout, stderr = run_iree_command(exec_args)
    if ret_value == 0:
        print(f"Successfully compiled {mlir_file} to {vmfb_file}")
        if stderr:
            with open(dump_file, "w") as f:
                f.write(stderr.decode("utf-8"))
    else:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {mlir_file}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(stderr.decode("utf-8"))
        return mlir_file, None

    return mlir_file, vmfb_file


# Dummy test generation
if __name__ == "__main__":
    config = AttentionConfig(20, 4096, 64, 64, 4096, "f16")
    spec = TuningSpec(
        [1, 128, 0, 0, 0],
        [0, 0, 0, 0, 32],
        4,
        1,
        IntrinsicType.VMFMA_F32_32x32x16_F16,
        2,
        True,
    )
    print(generate_mlir(config, spec))
