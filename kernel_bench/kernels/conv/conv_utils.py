from kernel_bench.config.types.conv import ConvConfig
from kernel_bench.utils.dtypes.device_context import DeviceContext
from kernel_bench.utils.iree_utils import shape_to_iree


def get_iree_conv_img_shape(config: ConvConfig, device_ctx: DeviceContext) -> str:
    """Get input image shape string."""
    if "nhwc" in config.OP:
        in_h = config.H * config.S + config.P - 1
        in_w = config.W * config.S + config.Q - 1
        return shape_to_iree(
            (config.N, in_h, in_w, config.C), config.input_dtype, device_ctx
        )
    if "nchw" in config.OP:
        in_h = config.H * config.S + config.P - 1
        in_w = config.W * config.S + config.Q - 1
        return shape_to_iree(
            (config.N, config.C, in_h, in_w), config.input_dtype, device_ctx
        )


def get_iree_conv_kernel_shape(config: ConvConfig, device_ctx: DeviceContext) -> str:
    """Get convolution kernel shape string."""
    if "nhwc" in config.OP:
        return shape_to_iree(
            (config.P, config.Q, config.C, config.F), config.input_dtype, device_ctx
        )
    if "nchw" in config.OP:
        return shape_to_iree(
            (config.F, config.C, config.P, config.Q), config.input_dtype, device_ctx
        )


def get_iree_conv_out_shape(config: ConvConfig, device_ctx: DeviceContext) -> str:
    """Get output shape string."""
    padding = 0
    in_h = config.H * config.S + config.P - 1
    in_w = config.W * config.S + config.Q - 1
    h_out = (in_h + 2 * padding - config.P) // config.S + 1
    w_out = (in_w + 2 * padding - config.Q) // config.S + 1
    n = config.N
    nf = config.F
    if "nhwc" in config.OP:
        return shape_to_iree((n, h_out, w_out, nf), config.output_dtype, device_ctx)
    if "nchw" in config.OP:
        return shape_to_iree((n, nf, h_out, w_out), config.output_dtype, device_ctx)
