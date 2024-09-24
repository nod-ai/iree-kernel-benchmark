module {
    func.func @main_0(%arg0: tensor<4096x8192xf16>, %arg1: tensor<8192x4096xf16>) -> tensor<4096x4096xf16> {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = tensor.empty() : tensor<4096x4096xf16>
        %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
        %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x8192xf16>, tensor<8192x4096xf16>) outs(%1 : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
        return %2 : tensor<4096x4096xf16>
    }
} 
