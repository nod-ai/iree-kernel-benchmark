from utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GemmConfig:
    M: int
    N: int
    K: int
    tA: str
    tB: str
    dtype: str

    def get_name(self) -> str:
        name = f"gemm_{self.M}_{self.N}_{self.K}_{self.dtype}"
        if self.tA == "T":
            name += "_tA"
        elif self.tB == "T":
            name += "_tB"
        return name

    def get_inp1(self) -> str:
        if self.tA == "T":
            inp1 = f"{self.K}x{self.M}x{self.dtype}"
        else:
            inp1 = f"{self.M}x{self.K}x{self.dtype}"
        return inp1

    def get_inp2(self) -> str:
        if self.tB == "T":
            inp2 = f"{self.N}x{self.K}x{self.dtype}"
        else:
            inp2 = f"{self.K}x{self.N}x{self.dtype}"
        return inp2

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
        element_count = self.M * self.K + self.N * self.K + self.M * self.N
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        flops = 2 * self.M * self.N * self.K
        return flops

def generate_mlir(config: GemmConfig):
    K = config.K
    M = config.M
    N = config.N
    dtype = config.dtype
    tA = config.tA
    tB = config.tB
    mlir_template_A = f"""
module {{
    func.func @main(%arg0: tensor<{K}x{M}x{dtype}>, %arg1: tensor<{K}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
        %cst = arith.constant 0.000000e+00 : {dtype}
        %0 = tensor.empty() : tensor<{M}x{N}x{dtype}>
        %1 = linalg.fill ins(%cst : {dtype}) outs(%0 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<{K}x{M}x{dtype}>, tensor<{K}x{N}x{dtype}>) outs(%1 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        return %2 : tensor<{M}x{N}x{dtype}>
    }}
}}
"""

    mlir_template_B = f"""
module {{
    func.func @main(%arg0: tensor<{M}x{K}x{dtype}>, %arg1: tensor<{N}x{K}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
        %cst = arith.constant 0.000000e+00 : {dtype}
        %0 = tensor.empty() : tensor<{M}x{N}x{dtype}>
        %1 = linalg.fill ins(%cst : {dtype}) outs(%0 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<{M}x{K}x{dtype}>, tensor<{N}x{K}x{dtype}>) outs(%1 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        return %2 : tensor<{M}x{N}x{dtype}>
    }}
}}
"""

    mlir_template = f"""module {{
    func.func @main(%arg0: tensor<{M}x{K}x{dtype}>, %arg1: tensor<{K}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
        %cst = arith.constant 0.000000e+00 : {dtype}
        %0 = tensor.empty() : tensor<{M}x{N}x{dtype}>
        %1 = linalg.fill ins(%cst : {dtype}) outs(%0 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        %2 = linalg.matmul ins(%arg0, %arg1 : tensor<{M}x{K}x{dtype}>, tensor<{K}x{N}x{dtype}>) outs(%1 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        return %2 : tensor<{M}x{N}x{dtype}>
    }}
}}
"""
    if tA == "T":
        return mlir_template_A
    if tB == "T":
        return mlir_template_B
    return mlir_template


def compile_gemm_config(
    config: GemmConfig, kernel_dir: Path, vmfb_dir: Path, target, extra_compiler_args
) -> tuple[Path, Optional[Path]]:
    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")

    if not os.path.exists(vmfb_dir):
        os.makedirs(vmfb_dir)

    # Generate mlir content
    mlir_content = generate_mlir(config)

    # Write MLIR content to file
    with open(mlir_file, "w") as f:
        f.write(mlir_content)

    # Compile MLIR to VMFB
    exec_args = [
        "iree-compile",
        f"{mlir_file}",
        "--iree-hal-target-backends=rocm",
        f"--iree-hip-target={target}",
        "--iree-llvmgpu-enable-prefetch=true",
        "-o",
        f"{vmfb_file}",
    ] + extra_compiler_args

    print(" ".join(exec_args))

    ret_value, stderr = run_iree_command(exec_args)
    if ret_value != 0:
        print(f"Successfully compiled {mlir_file} to {vmfb_file}")
    else:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {mlir_file}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(stderr.decode("utf-8"))
        return mlir_file, None

    return mlir_file, vmfb_file
