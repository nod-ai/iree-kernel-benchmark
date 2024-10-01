
module {
    func.func @main(%arg0: tensor<8192x57344xbf16>, %arg1: tensor<8192x32xbf16>) -> tensor<57344x32xbf16> {
        %cst = arith.constant 0.000000e+00 : bf16
        %0 = tensor.empty() : tensor<57344x32xbf16>
        %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<57344x32xbf16>) -> tensor<57344x32xbf16>
        %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<8192x57344xbf16>, tensor<8192x32xbf16>) outs(%1 : tensor<57344x32xbf16>) -> tensor<57344x32xbf16>
        return %2 : tensor<57344x32xbf16>
    }
}
