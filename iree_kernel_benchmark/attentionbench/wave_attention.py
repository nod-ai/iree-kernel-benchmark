from ..utils import *
from .attention_config import AttentionAttributes
from .attention_utils import (
    AttentionBMNKTuningSpec,
    AttentionBSHDTuningSpec,
)
from pathlib import Path
from typing import Optional

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.kernel.wave.templates.gqa_vanilla_attention import (
    get_gqa_bshd_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


def compile_attention_wave_vanilla(
    shape: AttentionAttributes,
    mlir_file: Path,
    vmfb_file: Path,
    spec: Optional[AttentionBMNKTuningSpec],
    dump_dir: Optional[Path],
    mfma_variant: tuple[MMAType] = (
        MMAType.F32_32x32x16_K8_F16,
        MMAType.F32_32x32x8_F16,
    ),
) -> tuple[Path, Optional[Path]]:

    base_attention, hyperparams, dynamic_symbols = get_vanilla_attention_kernel(
        shape=shape, mfma_variant=mfma_variant, dynamic_dims=False
    )

    if spec:
        hyperparams.update(spec.hyperparams())

    hyperparams.update(get_default_scheduling_params())

    config = shape.to_bmnk1k2()

    compile_options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        iree_launch_async=False,
        backend="rocm",
        target="gfx942",
        print_ir_after_all=dump_dir is not None,
    )

    if dump_dir:
        dump_file = dump_dir / "wave" / (config.get_name() + ".debug.mlir")
        with redirect_stderr_to_file(dump_file):
            result = wave_compile(compile_options, base_attention)
    else:
        result = wave_compile(compile_options, base_attention)

    with open(mlir_file, "w") as mlir_out:
        mlir_out.write(result.asm)

    return mlir_file, vmfb_file


def compile_attention_wave_bshd(
    shape: AttentionAttributes,
    mlir_file: Path,
    vmfb_file: Path,
    spec: Optional[AttentionBSHDTuningSpec],
    dump_dir: Optional[Path],
    mfma_variant: tuple[MMAType] = (
        MMAType.F32_32x32x16_K8_F16,
        MMAType.F32_32x32x8_F16,
    ),
) -> tuple[Path, Optional[Path]]:

    base_attention, hyperparams, dynamic_symbols = get_gqa_bshd_attention_kernel(
        shape=shape,
        mfma_variant=mfma_variant,
        input_dtype=dtype_to_torch(shape.dtype),
        output_dtype=dtype_to_torch("f32"),
    )

    if spec:
        hyperparams.update(spec.hyperparams())
    hyperparams.update(get_default_scheduling_params())

    config = shape.to_bshd()

    compile_options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        iree_launch_async=False,
        backend="rocm",
        target="gfx942",
        print_ir_after_all=dump_dir is not None,
    )

    if dump_dir:
        dump_file = dump_dir / "wave" / (config.get_name() + ".debug.mlir")
        with redirect_stderr_to_file(dump_file):
            result = wave_compile(compile_options, base_attention)
    else:
        result = wave_compile(compile_options, base_attention)

    with open(mlir_file, "w") as mlir_out:
        mlir_out.write(result.asm)

    return mlir_file, vmfb_file
