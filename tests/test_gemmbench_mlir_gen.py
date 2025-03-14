from iree_kernel_benchmark.gemmbench.gemm_utils import GemmConfig, generate_mlir
from .utils import match_lines


def test1():
    cfg = GemmConfig(
        M=512,
        N=4096,
        K=14336,
        tA="N",
        tB="T",
        operand_element_type="f16",
        accumulator_element_type="f32",
        result_element_type="f16",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<512x14336xf16>, %arg1: tensor<4096x14336xf16>) -> tensor<512x4096xf16> {",
            "%cst = arith.constant 0.0 : f32",
            "%0 = tensor.empty() : tensor<512x4096xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x4096xf32>) -> tensor<512x4096xf32>",
            "%2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<512x14336xf16>, tensor<4096x14336xf16>)",
            "outs(%1 : tensor<512x4096xf32>)",
            "-> tensor<512x4096xf32>",
            "%3 = arith.truncf %2 : tensor<512x4096xf32> to tensor<512x4096xf16>",
            "return %3 : tensor<512x4096xf16>",
        ],
    )
