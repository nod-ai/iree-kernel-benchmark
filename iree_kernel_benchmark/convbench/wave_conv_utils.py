from ..utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .conv_utils import ConvConfig
import traceback

try:
    import wave_lang.kernel as tk
    import wave_lang.kernel.lang as tkl
    from wave_lang.kernel.wave.templates.conv import get_igemm_conv2d
    from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
    from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
    from wave_lang.kernel.wave.utils.torch_utils import (
        device_randn,
        device_randint,
        device_randperm,
        device_zeros,
    )
except ImportError as e:
    TURBINE_AVAILABLE = False
    turbine_import_error = e
else:
    TURBINE_AVAILABLE = True


def compile_wave_conv_config(
    tag: str,
    config: ConvConfig,
    kernel_dir: Path,
    vmfb_dir: Path,
    extra_compiler_args: list[str],
) -> tuple[Path, Optional[Path]]:
    if not TURBINE_AVAILABLE:
        raise ValueError(
            f"Can't compile TK benchmark because of a failed import (most likely iree.turbine is missing): {turbine_import_error}"
        )

    # Name with tag is used for filenames so that duplicate configs with
    # different tags will not clobber eachother.
    name_with_tag = tag + "-" + config.get_name()
    mlir_file = kernel_dir / (name_with_tag + ".mlir")
    vmfb_file = vmfb_dir / (name_with_tag + ".vmfb")
    files_path = vmfb_dir / name_with_tag

    try:
        _compile_conv(config, mlir_file, vmfb_file)
    except Exception as e:
        error_file = vmfb_dir / (config.get_name() + "_error.txt")
        print(f"Failed to compile {config.get_name()}. Error dumped in {error_file}")
        with open(error_file, "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        return mlir_file, None, None

    return mlir_file, vmfb_file, files_path


def _decode_op(op: str) -> tuple[str, str]:
    if op.startswith("conv_2d_"):
        return "conv_2d", op[len("conv_2d_") :]

    raise ValueError(f"Unsupported op: {op}")


def _convert_dtype(dtype: str):
    dtypes = {
        "i8": tkl.i8,
        "i16": tkl.i16,
        "i32": tkl.i32,
        "i64": tkl.i64,
        "f16": tkl.f16,
        "f32": tkl.f32,
        "f64": tkl.f64,
        "bf16": tkl.bf16,
    }
    return dtypes[dtype]


def _compile_conv(config: ConvConfig, mlir_file: Path, vmfb_file: Path):
    op_type, layout = _decode_op(config.OP)

    in_h = config.H * config.S + config.P - 1
    in_w = config.W * config.S + config.Q - 1
    if op_type == "conv_2d":
        conv, hyperparams = get_igemm_conv2d(
            layout=layout,
            n=config.N,
            h=in_h,
            w=in_w,
            c=config.C,
            hf=config.P,
            wf=config.Q,
            nf=config.F,
            stride=config.S,
            input_dtype=_convert_dtype(config.input_dtype),
            output_dtype=_convert_dtype(config.output_dtype),
        )
    else:
        raise ValueError(f"Unsupported op_type: {op_type}")

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        schedule=SchedulingType.NONE,
        # inline=False, (TODO: how to do this with new API?)
        iree_launch_async=False,
        backend="rocm",
        target="gfx942",
    )
    result = wave_compile(options, conv)
    with open(mlir_file, "w") as f:
        f.write(result.asm)
