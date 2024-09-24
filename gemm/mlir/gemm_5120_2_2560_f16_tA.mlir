
module {
    func.func @main_0(%arg0: tensor<2560x5120xf16>, %arg1: tensor<2560x2xf16>) -> tensor<5120x2xf16> {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = tensor.empty() : tensor<5120x2xf16>
        %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<5120x2xf16>) -> tensor<5120x2xf16>
        %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<2560x5120xf16>, tensor<2560x2xf16>) outs(%1 : tensor<5120x2xf16>) -> tensor<5120x2xf16>
        return %2 : tensor<5120x2xf16>
    }
} 
