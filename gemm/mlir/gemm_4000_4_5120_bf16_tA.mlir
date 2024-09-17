
module {
    func.func @main_0(%arg0: tensor<4x4000xbf16>, %arg1: tensor<4x5120xbf16>) -> tensor<4000x5120xbf16> {
        %cst = arith.constant 0.000000e+00 : bf16
        %0 = tensor.empty() : tensor<4000x5120xbf16>
        %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4000x5120xbf16>) -> tensor<4000x5120xbf16>
        %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<4x4000xbf16>, tensor<4x5120xbf16>) outs(%1 : tensor<4000x5120xbf16>) -> tensor<4000x5120xbf16>
        return %2 : tensor<4000x5120xbf16>
    }
} 
