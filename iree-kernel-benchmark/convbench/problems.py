from .conv_utils import ConvConfig
import os

def unet_sweep(op: str, input_dtype: str, output_dtype: str) -> list[ConvConfig]:
    configs = []
    for B in [1, 2, 4, 8]:
        configs.append(ConvConfig(B, 128, 128, 16, 3, 3, 320, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 320, 3, 3, 320, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 320, 3, 3, 320, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 320, 3, 3, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 640, 3, 3, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 320, 1, 1, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 640, 3, 3, 640, 2, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 640, 3, 3, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 1280, 3, 3, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 640, 1, 1, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 2560, 3, 3, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 2560, 1, 1, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 1920, 3, 3, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 32, 32, 1920, 1, 1, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 1280, 3, 3, 1280, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 1920, 3, 3, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 1920, 1, 1, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 1280, 3, 3, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 1280, 1, 1, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 960, 3, 3, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 64, 64, 960, 1, 1, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 640, 3, 3, 640, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 960, 3, 3, 320, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 960, 1, 1, 320, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 640, 3, 3, 320, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 640, 1, 1, 320, 1, op, input_dtype, output_dtype))
        configs.append(ConvConfig(B, 128, 128, 320, 3, 3, 16, 1, op, input_dtype, output_dtype))
    return configs

def resnet_sweep(op: str, input_dtype: str, output_dtype: str) -> list[ConvConfig]:
    configs = []
    for B in [1, 2, 4, 8]:
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

    # Resnet
    resnet_configs = []
    resnet_configs += resnet_sweep("conv_2d_nhwc_hwcf", "f16", "f32")
    resnet_configs += resnet_sweep("conv_2d_nhwc_hwcf", "i8", "i32")
    resnet_configs += resnet_sweep("conv_2d_nchw_fchw", "f16", "f32")
    resnet_configs += resnet_sweep("conv_2d_nchw_fchw", "i8", "i32")
    configs += [("resnet", x) for x in resnet_configs]

    # Unet
    unet_configs = []
    unet_configs += unet_sweep("conv_2d_nhwc_hwcf", "f16", "f32")
    unet_configs += unet_sweep("conv_2d_nhwc_hwcf", "i8", "i32")
    unet_configs += unet_sweep("conv_2d_nchw_fchw", "f16", "f32")
    unet_configs += unet_sweep("conv_2d_nchw_fchw", "i8", "i32")
    configs += [("unet", x) for x in unet_configs]

    # MIOpen bf16 convs
    cwd = os.getcwd()
    shapes_file_path = os.path.join(cwd, "iree-kernel-benchmark/convbench/miopen_conv2d_shapes.txt")
    miopen_configs = []
    with open(shapes_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            params = line.split(", ")
            miopen_configs.append(ConvConfig(int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]), int(params[7]), params[8], params[9], params[10]))
    configs += [("miopen", x) for x in miopen_configs]

    return configs

def get_tk_conv_configs() -> list[tuple[str, ConvConfig]]:
    def check(config_tuple: tuple[str, ConvConfig]) -> bool:
        config = config_tuple[1]
        return config.input_dtype == "f16" and config.output_dtype == "f32"

    return list(filter(check, get_conv_configs()))


# Test function to run only a few chosen shapes
def get_conv_test_configs() -> list[tuple[str, ConvConfig]]:
    configs: list[tuple[str, ConvConfig]] = []

    resnet_configs = []
    # resnet_configs += resnet_sweep("conv_2d_nhwc_hwcf", "f16", "f32")
    # resnet_configs += resnet_sweep("conv_2d_nhwc_hwcf", "i8", "i32")
    # resnet_configs += resnet_sweep("conv_2d_nchw_fchw", "f16", "f32")
    # resnet_configs += resnet_sweep("conv_2d_nchw_fchw", "i8", "i32")
    configs += [("resnet", x) for x in resnet_configs]
    
    unet_configs = []
    # unet_configs.append(ConvConfig(1,128,128,16,3,3,320,1, "conv_2d_nhwc_hwcf_q", "i8", "i32"))
    # unet_configs.append(ConvConfig(1,32,32,640,1,1,1280,1, "conv_2d_nhwc_hwcf_q", "i8", "i32"))

    configs += [("unet", x) for x in unet_configs]

    return configs
