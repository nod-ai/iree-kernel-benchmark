from ..utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum


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
    return ["--iree-codegen-gpu-native-math-precision"]


def compile_attention_config(
    config: AttentionConfig, kernel_dir: Path, vmfb_dir: Path
) -> tuple[Path, Optional[Path]]:
    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")
    dump_file = kernel_dir / (config.get_name() + ".stderr.mlir")

    # TODO: Use different tuning specs for different configs. This is just a
    # general tuning config that worked well for sdxl shapes.
    spec = TuningSpec(
        [1, 128, 0, 0, 0],
        [0, 0, 0, 0, 32],
        4,
        1,
        IntrinsicType.VMFMA_F32_32x32x16_F16,
        2,
        True,
    )
    # Generate mlir content
    mlir_content = generate_mlir(config, spec)

    # Write MLIR content to file
    with open(mlir_file, "w") as f:
        f.write(mlir_content)

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
