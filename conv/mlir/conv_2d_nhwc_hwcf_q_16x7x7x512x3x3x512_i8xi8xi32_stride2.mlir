util.func public @main(%arg0: tensor<16x16x16x512xi8>, %arg1: tensor<3x3x512x512xi8>) -> tensor<16x7x7x512xi32> {
    %cst = arith.constant 0 : i32
    %9 = tensor.empty() : tensor<16x7x7x512xi32>
    %10 = linalg.fill ins(%cst : i32) outs(%9 : tensor<16x7x7x512xi32>) -> tensor<16x7x7x512xi32>
    %c0_i32 = arith.constant 0 : i32
    %11 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<16x16x16x512xi8>, tensor<3x3x512x512xi8>, i32, i32) outs(%10 : tensor<16x7x7x512xi32>) -> tensor<16x7x7x512xi32>
    util.return %11 : tensor<16x7x7x512xi32>
}
