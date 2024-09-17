
module {
    func.func @main_0(%arg0: tensor<32x10240xf16>, %arg1: tensor<32x8192xf16>) -> tensor<10240x8192xf16> {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = tensor.empty() : tensor<10240x8192xf16>
        %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<10240x8192xf16>) -> tensor<10240x8192xf16>
        %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<32x10240xf16>, tensor<32x8192xf16>) outs(%1 : tensor<10240x8192xf16>) -> tensor<10240x8192xf16>
        return %2 : tensor<10240x8192xf16>
    }
} 
