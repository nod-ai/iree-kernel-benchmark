from iree_kernel_benchmark.gemmbench.gemm_utils import GemmConfig, generate_mlir
from .utils import match_lines
from iree.compiler import ir
import pytest

# These tests should contain a small sampling of the actual problem set, enough
# to exercise most of the code paths in the MLIR generation.


@pytest.fixture(autouse=True)
def run_with_mlir_ctx():
    with ir.Context():
        yield


def test_n_t_f16_f32_f16():
    # From 'llama8b_prefill'
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
            "%cst = arith.constant 0.000000e+00 : f32",
            "%0 = tensor.empty() : tensor<512x4096xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x4096xf32>) -> tensor<512x4096xf32>",
            "%2 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<512x14336xf16>, tensor<4096x14336xf16>) outs(%1 : tensor<512x4096xf32>) -> tensor<512x4096xf32>",
            "%3 = arith.truncf %2 : tensor<512x4096xf32> to tensor<512x4096xf16>",
            "return %3 : tensor<512x4096xf16>",
        ],
    )


def test_n_t_f8_f32_f8():
    # From 'llama8b_prefill' (f8 version is synthetic)
    cfg = GemmConfig(
        M=512,
        N=4096,
        K=14336,
        tA="N",
        tB="T",
        operand_element_type="f8E4M3FNUZ",
        accumulator_element_type="f32",
        result_element_type="f8E4M3FNUZ",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<512x14336xf8E4M3FNUZ>, %arg1: tensor<4096x14336xf8E4M3FNUZ>) -> tensor<512x4096xf8E4M3FNUZ> {",
            "%cst = arith.constant 0.000000e+00 : f32",
            "%0 = tensor.empty() : tensor<512x4096xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x4096xf32>) -> tensor<512x4096xf32>",
            "%2 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<512x14336xf8E4M3FNUZ>, tensor<4096x14336xf8E4M3FNUZ>) outs(%1 : tensor<512x4096xf32>) -> tensor<512x4096xf32>",
            "%3 = arith.truncf %2 : tensor<512x4096xf32> to tensor<512x4096xf8E4M3FNUZ>",
            "return %3 : tensor<512x4096xf8E4M3FNUZ>",
        ],
    )


def test_n_t_bf16_f32_bf16():
    # From 'llama70bmemory'
    cfg = GemmConfig(
        M=2,
        N=1280,
        K=8192,
        tA="N",
        tB="T",
        operand_element_type="bf16",
        accumulator_element_type="f32",
        result_element_type="bf16",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<2x8192xbf16>, %arg1: tensor<1280x8192xbf16>) -> tensor<2x1280xbf16> {",
            "%cst = arith.constant 0.000000e+00 : f32",
            "%0 = tensor.empty() : tensor<2x1280xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x1280xf32>) -> tensor<2x1280xf32>",
            "%2 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<2x8192xbf16>, tensor<1280x8192xbf16>) outs(%1 : tensor<2x1280xf32>) -> tensor<2x1280xf32>",
            "%3 = arith.truncf %2 : tensor<2x1280xf32> to tensor<2x1280xbf16>",
            "return %3 : tensor<2x1280xbf16>",
        ],
    )


def test_t_n_f16_f32_f16():
    # From 'llama13bmatvec'
    cfg = GemmConfig(
        M=32000,
        N=1,
        K=5120,
        tA="T",
        tB="N",
        operand_element_type="f16",
        accumulator_element_type="f32",
        result_element_type="f16",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<5120x32000xf16>, %arg1: tensor<5120x1xf16>) -> tensor<32000x1xf16> {",
            "%cst = arith.constant 0.000000e+00 : f32",
            "%0 = tensor.empty() : tensor<32000x1xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32000x1xf32>) -> tensor<32000x1xf32>",
            "%2 = linalg.matmul_transpose_a {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<5120x32000xf16>, tensor<5120x1xf16>) outs(%1 : tensor<32000x1xf32>) -> tensor<32000x1xf32>",
            "%3 = arith.truncf %2 : tensor<32000x1xf32> to tensor<32000x1xf16>",
            "return %3 : tensor<32000x1xf16>",
        ],
    )


def test_t_n_bf16_f32_bf16():
    # From 'llama13bmatvec'
    cfg = GemmConfig(
        M=32000,
        N=1,
        K=5120,
        tA="T",
        tB="N",
        operand_element_type="bf16",
        accumulator_element_type="f32",
        result_element_type="bf16",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<5120x32000xbf16>, %arg1: tensor<5120x1xbf16>) -> tensor<32000x1xbf16> {",
            "%cst = arith.constant 0.000000e+00 : f32",
            "%0 = tensor.empty() : tensor<32000x1xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32000x1xf32>) -> tensor<32000x1xf32>",
            "%2 = linalg.matmul_transpose_a {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<5120x32000xbf16>, tensor<5120x1xbf16>) outs(%1 : tensor<32000x1xf32>) -> tensor<32000x1xf32>",
            "%3 = arith.truncf %2 : tensor<32000x1xf32> to tensor<32000x1xbf16>",
            "return %3 : tensor<32000x1xbf16>",
        ],
    )


def test_n_n_f16_f32_f16():
    # From 'gpt4compute'
    cfg = GemmConfig(
        M=2048,
        N=2048,
        K=1024,
        tA="N",
        tB="N",
        operand_element_type="f16",
        accumulator_element_type="f32",
        result_element_type="f16",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<2048x1024xf16>, %arg1: tensor<1024x2048xf16>) -> tensor<2048x2048xf16> {",
            "%cst = arith.constant 0.000000e+00 : f32",
            "%0 = tensor.empty() : tensor<2048x2048xf32>",
            "%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>",
            "%2 = linalg.matmul ins(%arg0, %arg1 : tensor<2048x1024xf16>, tensor<1024x2048xf16>) outs(%1 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>",
            "%3 = arith.truncf %2 : tensor<2048x2048xf32> to tensor<2048x2048xf16>",
            "return %3 : tensor<2048x2048xf16>",
        ],
    )


def test_n_t_i8_i32_i8():
    # From 'square'
    cfg = GemmConfig(
        M=128,
        N=128,
        K=128,
        tA="N",
        tB="T",
        operand_element_type="i8",
        accumulator_element_type="i32",
        result_element_type="i8",
    )
    mlir = generate_mlir(cfg)
    match_lines(
        mlir,
        [
            "module {",
            "func.func @main(%arg0: tensor<128x128xi8>, %arg1: tensor<128x128xi8>) -> tensor<128x128xi8> {",
            "%c0_i32 = arith.constant 0 : i32",
            "%0 = tensor.empty() : tensor<128x128xi32>",
            "%1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>",
            "%2 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<128x128xi8>, tensor<128x128xi8>) outs(%1 : tensor<128x128xi32>) -> tensor<128x128xi32>",
            "%3 = arith.trunci %2 : tensor<128x128xi32> to tensor<128x128xi8>",
            "return %3 : tensor<128x128xi8>",
        ],
    )
