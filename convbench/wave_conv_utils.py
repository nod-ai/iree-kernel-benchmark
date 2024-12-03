from utils import *
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from conv_utils import ConvConfig
import traceback

try:
    import iree.turbine.kernel as tk
    import iree.turbine.kernel.lang as tkl
    from iree.turbine.kernel.wave.templates.conv import get_igemm_conv2d
    from iree.turbine.kernel.wave.utils import (
        get_default_arch,
        get_default_run_config,
        get_default_compile_config,
        device_randn,
        device_randint,
        device_randperm,
        device_zeros,
    )
except ImportError:
    TURBINE_AVAILABLE=False
else:
    TURBINE_AVAILABLE=True

def compile_wave_conv_config(
    config: ConvConfig, kernel_dir: Path, vmfb_dir: Path, extra_compiler_args: list[str]
) -> tuple[Path, Optional[Path]]:
    if not TURBINE_AVAILABLE:
        raise ValueError("iree.turbine package is not available")

    mlir_file = kernel_dir / (config.get_name() + ".mlir")
    vmfb_file = vmfb_dir / (config.get_name() + ".vmfb")
    # dump_file = kernel_dir / (config.get_name() + ".stderr.mlir")
    files_path = vmfb_dir / config.get_name()

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
        return "conv_2d", op[len("conv_2d_"):]

    raise ValueError(f"Unsupported op: {op}")

def _convert_dtype(dtype:str):
    dtypes = {
        "i8": tkl.i8,
        "i16": tkl.i16,
        "i32": tkl.i32,
        "i64": tkl.i64,
        "f16": tkl.f16,
        "f32": tkl.f32,
        "f64": tkl.f64,
        # "bf16": tkl.bf16, TODO
    }
    return dtypes[dtype]

def _compile_conv(config: ConvConfig, mlir_file: Path, vmfb_file: Path):
    print("Compile TKW kernel", config.OP)
    op_type, layout = _decode_op(config.OP)

    if op_type == "conv_2d":
        conv, hyperparams = get_igemm_conv2d(
            layout=layout,
            n=config.N,
            h=config.H,
            w=config.W,
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

    # config = get_default_run_config()
    config = get_default_compile_config()

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        run_config=config,
        schedule=False,
    ):
        mod = conv().module_op # This will generate vmfb file
        with open(mlir_file, "w") as f:
            f.write(str(mod))

        print(f"Successfully compiled to {vmfb_file}")

