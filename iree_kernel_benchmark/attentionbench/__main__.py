import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from typing import Literal
from pathlib import Path
import argparse
import sys
from ..utils import *
from .attention_utils import *
from .attention_config import *
from .wave_attention import compile_attention_wave_vanilla
from .iree_attention import compile_attention_iree
from .problems import get_attention_configs

type Backend = Literal["iree", "wave"]

def compile_attention(
    tag: str, 
    config: AttentionAttributes, 
    kernel_dir: Path, 
    vmfb_dir: Path, 
    backend: Backend, 
    extra_compiler_args: list[str] = [], 
    dump_dir=None,
    mfma_variant: tuple[MMAType] = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
):
    name = config.to_bmnk1k2().get_name()
    if dump_dir:
        dump_dir = Path(dump_dir)
        dpath = dump_dir / name
        extra_compiler_args.extend([f"--iree-hal-dump-executable-files-to={dpath}"])
    
    mlir_file = kernel_dir / f"{name}.mlir"
    vmfb_file = vmfb_dir / f"{name}.vmfb"
    
    spec = TuningSpec(
        [1, 128, 0, 0, 0],
        [0, 0, 0, 0, 32],
        4,
        1,
        IntrinsicType.VMFMA_F32_32x32x16_F16,
        2,
        True,
    )

    if backend == "iree":
        mlir_file, vmfb_file = compile_attention_iree(config, spec, mlir_file, vmfb_file, dump_dir, extra_compiler_args)
    elif backend == "wave":
        mlir_file, vmfb_file = compile_attention_wave_vanilla(config, spec, mlir_file, vmfb_file, dump_dir, mfma_variant)
    
    return (tag, config, mlir_file, vmfb_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level",
    )
    parser.add_argument(
        "--roofline",
        help="Comma seperated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument(
        "--device",
        help="The IREE device to execute benchmarks on",
        type=str,
        default="hip",
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--backend",
        choices=["iree", "wave"],
        default="iree",
        help="Backend to run kernels",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="Directory to which executable files will be dumped.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline, args.plot, args.batch, args.dtype, args.model)
        sys.exit()
    
    # mfma_configs = [
    #     (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
    #     (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
    #     (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    #     (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    # ]

    mfma_config = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x16_K8_F16)

    configs = get_attention_configs()
    print(f"Generated {len(configs)} attention configs.")

    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()

    backend_name = args.backend

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "attention" / backend_name / "mlir"
    vmfb_dir = repo_root / "attention" / backend_name / "vmfb"
    dump_dir = args.dump_dir
    device = args.device
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)

    compile_args = itertools.starmap(
        lambda tag, config: (tag, config, kernel_dir, vmfb_dir, backend_name, [], dump_dir, mfma_config), configs
    )
    with Pool(num_cpus) as pool:
        compilation_results = list(
            tqdm(pool.istarmap(compile_attention, compile_args),
                 total=len(configs), desc="Compiling Attention Kernels")
        )

    error_count = 0
    for tag, config, mlir_file, vmfb_file in compilation_results:
        if vmfb_file:
            vmfb_dict[vmfb_file] = (tag, config)
        else:
            error_count += 1
    print(
        f"{len(configs) - error_count} Success, {error_count} Failed out of {len(configs)} configs"
    )

    print("Compilation process completed.")

    results = []
    index = 0

    csv_base_name = 'tiling/test_refactor'
    output_csv = f"results/{csv_base_name}_wave.csv" if backend_name == "wave" else f"results/{csv_base_name}_iree.csv"
    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for vmfb_filename, value in tqdm(vmfb_dict.items(), desc="Benchmarking Attention Kernels"):
        tag = value[0]
        attn_attrs : AttentionAttributes = value[1]
        config : AttentionConfigBMNK = attn_attrs.to_bmnk1k2()
        name = config.get_name()

        query_shape = config.get_query_shape()
        key_shape = config.get_key_shape()
        value_shape = config.get_value_shape()

        exec_args = [
            "iree-benchmark-module",
            f"--device={device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            "--function=main",
            f"--input={query_shape}",
            f"--input={key_shape}",
            f"--input={value_shape}",
            f"--benchmark_repetitions={args.iterations}",
        ]

        if backend_name == "wave":
            out_shape : str = config.get_output_shape()
            out_shape = 'x'.join(out_shape.split('x')[:-1] + ['f32'])
            exec_args.append(f"--input={out_shape}")
            exec_args += ["--function=isolated_benchmark"]
        elif backend_name == "iree":
            exec_args += ["--function=main"]

        # iree benchmark kernels
        ret_value, cmd_out, cmd_err = run_iree_command(exec_args)
        ok = ret_value == 0
        benchmark_gemm_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_gemm_mean_time_us = benchmark_gemm_mean_time_ms * 1000

        flops = config.get_flops()
        byte_count = config.get_byte_count()

        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_gemm_mean_time_us / 1e6)

        results.append(
            (
                index,
                tag,
                name,
                config.B,
                config.M,
                config.N,
                config.K1,
                config.K2,
                config.dtype,
                round(benchmark_gemm_mean_time_us, 4),
                round(arithmetic_intensity, 4),
                round(tflops_per_second, 4),
                ok,
            )
        )
        index += 1

    fieldnames = [
        "index",
        "tag",
        "name",
        "B",
        "M",
        "N",
        "K1",
        "K2",
        "dtype",
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "ok",
    ]

    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")
