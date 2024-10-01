
module {
    func.func @main(%arg0: tensor<2048x5120xf16>, %arg1: tensor<1280x5120xf16>) -> tensor<2048x1280xf16> {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = tensor.empty() : tensor<2048x1280xf16>
        %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<2048x1280xf16>) -> tensor<2048x1280xf16>
        %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<2048x5120xf16>, tensor<1280x5120xf16>) outs(%1 : tensor<2048x1280xf16>) -> tensor<2048x1280xf16>
        return %2 : tensor<2048x1280xf16>
    }
}
