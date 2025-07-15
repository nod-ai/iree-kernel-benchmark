from ..utils import *
from .attention_config import AttentionAttributes, AttentionConfigBMNK
from .attention_utils import TuningSpec, IntrinsicType
from pathlib import Path
from typing import Optional

from iree.turbine.kernel.wave.constraints import MMAType
from typing import Optional


def generate_attention_mlir_iree(
    config: AttentionConfigBMNK, tuning: Optional[TuningSpec] = None
):
    shapes = f"""\
!dtype = {config.dtype}
!Q     = tensor<{config.get_query_shape()}>
!K     = tensor<{config.get_key_shape()}>
!V     = tensor<{config.get_value_shape()}>
!O     = tensor<{config.get_output_shape()}>
"""

    spec = ""
    if tuning and config.dtype == "f16":
        spec = f"""\
#tuning = {tuning.get_compilation_info()}
"""

    attn_kernel = f"""
#Q = affine_map<(b, m, n, k1, k2) -> (b, m, k1)>
#K = affine_map<(b, m, n, k1, k2) -> (b, k2, k1)>
#V = affine_map<(b, m, n, k1, k2) -> (b, k2, n)>
#S = affine_map<(b, m, n, k1, k2) -> ()>
#O = affine_map<(b, m, n, k1, k2) -> (b, m, n)>

func.func @main(%Q : !Q, %K : !K, %V : !V) -> !O {{
  %scale = arith.constant 1.0 : !dtype
  %empty = tensor.empty() : !O
  %O = iree_linalg_ext.attention
       {{ indexing_maps = [#Q, #K, #V, #S, #O]
          ,decomposition_config = {{
           qk_attrs = {{attention_qk_matmul, lowering_config = {tuning.get_qk_config_info()}}},
           pv_attrs = {{attention_pv_matmul, lowering_config = {tuning.get_pv_config_info()}}}
         }}
         {",compilation_info = #tuning" if tuning and config.dtype == "f16" else ""}
       }}
       ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype) outs(%empty : !O) {{
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
        }} -> !O
  return %O : !O
}}
"""
    mlir_template = shapes + "\n" + spec + "\n" + attn_kernel
    return mlir_template


def compile_attention_iree(
    shape: AttentionAttributes,
    spec: TuningSpec,
    mlir_file: Path,
    vmfb_file: Path,
    dump_dir: Path = None,
    extra_compiler_args: list[str] = [],
) -> tuple[Path, Optional[Path]]:
    config = shape.to_bmnk1k2()

    # TODO: Use different tuning specs for different configs. This is just a
    # general tuning config that worked well for sdxl shapes.

    mlir_content = generate_attention_mlir_iree(config, spec)

    # Write MLIR content to file
    with open(mlir_file, "w") as f:
        f.write(mlir_content)

    # TODO: Do not hardcode device information, instead pass it as a class
    # Compile MLIR to vmfb
    exec_args = [
        "iree-compile",
        # Input file
        f"{mlir_file}",
        # Output file
        "-o",
        f"{vmfb_file}",
        # Target Device: hip
        "--iree-hal-target-device=hip",
        # Device: MI300x
        "--iree-hip-target=gfx942",
        "--mlir-print-ir-after-all",
    ] + extra_compiler_args
    if dump_dir:
        dump_file = dump_dir / "iree" / (config.get_name() + ".debug.mlir")
        phase_dump = dump_dir / "iree" / config.get_name()
        exec_args.append(f"--dump-compilation-phases-to={phase_dump}")

    ret_value, stdout, stderr = run_iree_command(exec_args)

    if ret_value == 0:
        if stderr and dump_dir:
            with open(dump_file, "w") as f:
                f.write(stderr.decode("utf-8"))
    else:
        if dump_dir:
            error_file = dump_dir / "iree" / "log" / (config.get_name() + "_error.txt")
            print(f"Failed to compile {mlir_file}. Error dumped in {error_file}")
            with open(error_file, "w") as f:
                f.write(stderr.decode("utf-8"))
        else:
            print(f"Failed to compile {mlir_file}. No dump directory specified.")

        return mlir_file, None

    return mlir_file, vmfb_file
