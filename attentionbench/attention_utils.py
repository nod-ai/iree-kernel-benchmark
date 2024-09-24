from utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    M_warp: int
    N_warp: int
    intrinsic: str
    waves_per_eu: Optional[int]
    denorm_flush: bool

    def get_lowering_config(self) -> str:
        return (
            f"#iree_codegen.lowering_config<"
            + f"tile_sizes = [[{','.join([str(x) for x in self.wg_tiles])}]]"
            + f">"
        )

    def get_mma_schedule(self) -> str:
        return (
            f"#iree_gpu.mma_schedule<"
            + f"intrinsic = #iree_gpu.mma_layout<{self.intrinsic}>"
            + f", subgroup_m_count = {self.M_warp}"
            + f", subgroup_n_count = {self.N_warp}"
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
            + f"LLVMGPUVectorDistribute"
            + f" workgroup_size = [{self.N_warp * 64}, {self.M_warp}]"
            + f" subgroup_size = 64"
            + f" ,{{mma_schedule = {self.get_mma_schedule()}"
            + f" , llvm_func_attrs = {{ {','.join(llvm_func_attrs)} }}"
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


def generate_mlir(config: AttentionConfig, tuning: Optional[TuningSpec] = None):
    shapes = f"""\
!dtype = {config.dtype}
!Q     = tensor<{config.get_query_shape()}>
!K     = tensor<{config.get_key_shape()}>
!V     = tensor<{config.get_value_shape()}>
!O     = tensor<{config.get_output_shape()}>
"""

    spec = ""
    if tuning:
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
         {",compilation_info = #tuning" if tuning else ""}
       }}
       ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype)
       outs(%empty : !O) -> !O
  return %O : !O
}}
"""
    mlir_template = shapes + "\n" + spec + "\n" + attn_kernel
    return mlir_template


def get_attention_flags() -> list[str]:
    return []


def compile_attention_config(
    config: AttentionConfig, kernel_dir: Path, vmfb_dir: Path
) -> tuple[Path, Optional[Path]]:
    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")

    # TODO: Use different tuning specs for different configs. This is just a
    # general tuning config that worked well for sdxl shapes.
    spec = TuningSpec([1, 128, 0, 0, 32], 4, 1, "MFMA_F32_32x32x8_F16", 2, True)
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

    ret_value, stderr = run_iree_command(exec_args)
    if ret_value == 0:
        print(f"Successfully compiled {mlir_file} to {vmfb_file}")
    else:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {mlir_file}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(stderr.decode("utf-8"))
        return mlir_file, None

    return mlir_file, vmfb_file
