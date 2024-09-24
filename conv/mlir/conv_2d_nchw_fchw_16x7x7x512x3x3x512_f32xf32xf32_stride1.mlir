util.func public @main(%arg0: tensor<16x512x9x9xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<16x512x7x7xf32> {
    %cst = arith.constant 0.0 : f32
    %9 = tensor.empty() : tensor<16x512x7x7xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<16x512x7x7xf32>) -> tensor<16x512x7x7xf32>
    %11 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<16x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%10 : tensor<16x512x7x7xf32>) -> tensor<16x512x7x7xf32>
    util.return %11 : tensor<16x512x7x7xf32>
}
