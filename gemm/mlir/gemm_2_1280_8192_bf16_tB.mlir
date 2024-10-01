
module {
    func.func @main(%arg0: tensor<2x8192xbf16>, %arg1: tensor<1280x8192xbf16>) -> tensor<2x1280xbf16> {
        %cst = arith.constant 0.000000e+00 : bf16
        %0 = tensor.empty() : tensor<2x1280xbf16>
        %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<2x1280xbf16>) -> tensor<2x1280xbf16>
        %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<2x8192xbf16>, tensor<1280x8192xbf16>) outs(%1 : tensor<2x1280xbf16>) -> tensor<2x1280xbf16>
        return %2 : tensor<2x1280xbf16>
    }
}
