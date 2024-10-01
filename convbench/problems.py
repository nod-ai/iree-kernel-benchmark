from conv_utils import ConvConfig


def resnet_sweep(op: str, input_dtype: str, output_dtype: str) -> list[ConvConfig]:
    configs = []
    for B in [1, 2, 4, 8, 16, 32, 48]:
        configs.append(ConvConfig(B, 112, 112, 64, 7, 7, 3, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 56, 56, 64, 3, 3, 64, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 28, 28, 128, 3, 3, 128, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 28, 28, 512, 1, 1, 256, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 28, 28, 128, 3, 3, 128, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 14, 14, 256, 3, 3, 256, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 14, 14, 1024, 1, 1, 512, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 14, 14, 256, 3, 3, 256, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 7, 7, 512, 3, 3, 512, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 7, 7, 2048, 1, 1, 1024, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 7, 7, 512, 3, 3, 512, 1, op, input_dtype, output_dtype))   
    return configs

def get_conv_configs() -> list[tuple[str, ConvConfig]]:
    configs: list[tuple[str, ConvConfig]] = []
    resnet_configs = resnet_sweep("conv_2d_nchw_fchw", "f32", "f32")
    resnet_configs += resnet_sweep("conv_2d_nhwc_hwcf_q", "i8", "i32")

    configs += [("resnet_sweep", x) for x in resnet_configs]

    return configs
