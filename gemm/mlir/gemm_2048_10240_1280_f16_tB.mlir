
module {
    func.func @main_0(%arg0: tensor<2048x1280xf16>, %arg1: tensor<10240x1280xf16>) -> tensor<2048x10240xf16> {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = tensor.empty() : tensor<2048x10240xf16>
        %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<2048x10240xf16>) -> tensor<2048x10240xf16>
        %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<2048x1280xf16>, tensor<10240x1280xf16>) outs(%1 : tensor<2048x10240xf16>) -> tensor<2048x10240xf16>
        return %2 : tensor<2048x10240xf16>
    }
} 
