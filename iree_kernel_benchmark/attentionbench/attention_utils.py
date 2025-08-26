from ..utils import *
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from typing import Optional
from sympy import symbols
from wave_lang.kernel.wave.constraints import MMAType


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
            return f"#iree_gpu.virtual_mma_layout<{intrinsic.name}>"
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
class IREEAttentionTuningSpec:
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


@dataclass
class AttentionBMNKTuningSpec(TuningSpec):
    BLOCK_B: int = 1
    BLOCK_M: int = 128
    BLOCK_N: int = 64
    BLOCK_K2: int = 64

    mfma_variant: Tuple[MMAType]

    def to_dict(self):
        return {
            "BLOCK_B": self.BLOCK_B,
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "BLOCK_K2": self.BLOCK_K2,
            "mfma_variant": [mfma_type.name for mfma_type in self.mfma_variant],
        }

    def load_from_dict(self, obj):
        self.BLOCK_B = obj["BLOCK_B"]
        self.BLOCK_M = obj["BLOCK_M"]
        self.BLOCK_N = obj["BLOCK_N"]
        self.BLOCK_K2 = obj["BLOCK_K2"]
        self.mfma_variant = tuple(
            MMAType[mfma_name] for mfma_name in obj["mfma_variant"]
        )


@dataclass
class AttentionBSHDTuningSpec(TuningSpec):
    BLOCK_B: int = 1
    BLOCK_H: int = 1
    BLOCK_N_Q: int = 128
    BLOCK_D_KV: int = 64
    BLOCK_N_KV: int = 64

    mfma_variant: Tuple[MMAType]

    def to_dict(self):
        return {
            "BLOCK_B": self.BLOCK_B,
            "BLOCK_H": self.BLOCK_H,
            "BLOCK_N_Q": self.BLOCK_N_Q,
            "BLOCK_D_KV": self.BLOCK_D_KV,
            "BLOCK_N_KV": self.BLOCK_N_KV,
            "mfma_variant": [mfma_type.name for mfma_type in self.mfma_variant],
        }

    def load_from_dict(self, obj):
        self.BLOCK_B = obj["BLOCK_B"]
        self.BLOCK_H = obj["BLOCK_H"]
        self.BLOCK_N_Q = obj["BLOCK_N_Q"]
        self.BLOCK_D_KV = obj["BLOCK_D_KV"]
        self.BLOCK_N_KV = obj["BLOCK_N_KV"]
        self.mfma_variant = tuple(
            MMAType[mfma_name] for mfma_name in obj["mfma_variant"]
        )
