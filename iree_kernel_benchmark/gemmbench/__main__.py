# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import argparse
import sys
from ..utils import *
from .gemm_utils import *
from .iree_gemm import compile_iree_gemm
from .wave_gemm import compile_wave_gemm
from .torch_gemm import benchmark_torch_gemm
from .problems import get_gemm_configs, get_tk_gemm_configs, get_matching_configs


def compile_gemm(
    backend: str,
    tag: str,
    config: GemmConfig,
    kernel_dir: Path,
    vmfb_dir: Path,
    target: Optional[str] = None,
    extra_compiler_args: list[str] = [],
    dump_dir: Optional[Path] = None,
) -> tuple[str, GemmConfig, Path, Optional[Path]]:
    name = config.get_name()
    if dump_dir:
        dump_dir = Path(dump_dir)
        dpath = dump_dir / name
        extra_compiler_args.extend([f"--iree-hal-dump-executable-files-to={dpath}"])

    mlir_file = kernel_dir / f"{name}.mlir"
    vmfb_file = vmfb_dir / f"{name}.vmfb"

    if backend == "iree":
        mlir_file, vmfb_file = compile_iree_gemm(
            config, kernel_dir, vmfb_dir, target, extra_compiler_args
        )
    else:
        mlir_file, vmfb_file = compile_wave_gemm(config, mlir_file, vmfb_file, dump_dir)

    return (tag, config, mlir_file, vmfb_file)


def compile_gemm_kernels(
    backend_name: str,
    configs: List[GemmConfig],
    kernel_dir: Path,
    vmfb_dir: Path,
    dump_dir: Optional[Path],
    target: str = None,
    extra_compiler_args: list[str] = [],
) -> dict[str, tuple[str, GemmConfig]]:
    def compile_args_generator():
        return itertools.starmap(
            lambda tag, config: (
                backend_name,
                tag,
                config,
                kernel_dir,
                vmfb_dir,
                target,
                extra_compiler_args,
                dump_dir,
            ),
            configs,
        )

    num_cpus = max(1, max(cpu_count() // 2, 1))
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    shared_vmfb_dict = manager.dict()

    with Pool(num_cpus) as pool:
        compilation_results = list(
            tqdm(
                pool.istarmap(compile_gemm, compile_args_generator()),
                total=len(configs),
                desc="Compiling GEMM Kernels",
            )
        )
    vmfb_dict = shared_vmfb_dict

    compile_error_count = 0
    for tag, config, mlir_file, vmfb_file in compilation_results:
        if vmfb_file:
            vmfb_dict[vmfb_file] = (tag, config)
        else:
            compile_error_count += 1
    print(
        f"{len(configs) - compile_error_count} Success, {compile_error_count} Failed out of {len(configs)} configs"
    )
    print("Compilation process completed.")

    return dict(vmfb_dict)


def save_results(
    configs: List[Tuple[str, GemmConfig]],
    runtimes_us: List[float],
    ok: List[bool],
    output_csv: Path,
):

    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    results = []
    index = 0

    for i, (tag, config) in enumerate(configs):
        flops = config.get_flops()
        byte_count = config.get_byte_count()

        benchmark_mean_time_us = runtimes_us[i]

        arithmetic_intensity = flops / byte_count
        if benchmark_mean_time_us == 0:
            tflops_per_second = 0
        else:
            tflops_per_second = (flops / 1e12) / (benchmark_mean_time_us / 1e6)

        results.append(
            (
                index,
                tag,
                config.get_name(),
                config.M,
                config.N,
                config.K,
                config.operand_element_type,
                config.tA,
                config.tB,
                f"D={config.runtime_dim}" if config.runtime_dim is not None else "",
                round(benchmark_mean_time_us, 4),
                round(arithmetic_intensity, 4),
                round(tflops_per_second, 4),
                ok[i],
            )
        )
        index += 1

    fieldnames = [
        "index",
        "tag",
        "name",
        "M",
        "N",
        "K",
        "dtype",
        "tA",
        "tB",
        "runtime_dim",
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "ok",
    ]

    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")


def benchmark_gemm_kernels(
    backend_name: str,
    vmfb_dict: dict[str, tuple[str, GemmConfig]],
    output_csv: Path,
    num_iterations: int = 3,
    debug=True,
):
    configs = []
    runtimes = []
    statuses = []

    run_error_count = 0
    for vmfb_filename, value in tqdm(
        vmfb_dict.items(), desc="Benchmarking GEMM Kernels"
    ):
        tag, config = value
        name = config.get_name()

        inp1 = config.get_inp1()
        inp2 = config.get_inp2()

        exec_args = [
            "iree-benchmark-module",
            f"--device={device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            f"--benchmark_repetitions={num_iterations}",
            f"--input={inp1}",
            f"--input={inp2}",
        ]

        if backend_name == "wave":
            out_shape = config.get_out()
            out_shape = "x".join(out_shape.split("x")[:-1] + ["f32"])
            exec_args.append(f"--input={out_shape}")
            exec_args += ["--function=isolated_benchmark"]
        else:
            exec_args += ["--function=main"]

        # iree benchmark kernels
        ret_value, cmd_out, cmd_err = run_iree_command(exec_args)
        ok = ret_value == 0
        if not ok:
            run_error_count += 1
        benchmark_gemm_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        if benchmark_gemm_mean_time_ms is None:
            print(f"{name} benchmark failed. Skipping")
            continue
        benchmark_mean_time_us = benchmark_gemm_mean_time_ms * 1000

        configs.append((tag, config))
        runtimes.append(benchmark_mean_time_us)
        statuses.append(ok)

    save_results(configs, runtimes, statuses, output_csv)


def benchmark_extern_gemm_kernels(
    backend: str,
    configs: List[Tuple[str, GemmConfig]],
    output_csv: Path,
    num_iterations: int = 3,
):
    runtimes_us = []
    statuses = []

    for tag, config in tqdm(configs, f"Benchmarking {backend} gemm kernels"):
        if backend == "torch":
            mean_us = benchmark_torch_gemm(config, num_iterations)
            if mean_us:
                runtimes_us.append(mean_us)
                statuses.append(True)
            else:
                runtimes_us.append(0)
                statuses.append(False)

    return save_results(configs, runtimes_us, statuses, output_csv)


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
        "--target",
        help="The IREE hip target to compile for. The special value host_cpu results in a llvm-cpu benchmark instead of HIP, compiled for the host CPU.",
        type=str,
        default="gfx942",
    )
    parser.add_argument(
        "--device",
        help="The IREE device to execute benchmarks on",
        type=str,
        default="hip",
    )
    parser.add_argument(
        "--Xiree_compile",
        nargs="+",
        default=[],
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=[],
        help="List of data types to generate benchmarks for. Defaults to f16. Other options include (for example) f32, bf16, i8, f8E4M3FNUZ.",
    )
    parser.add_argument(
        "--raw_accumulators",
        action="store_true",
        help="If true, generate benchmark matmuls returning the raw accumulator type with no truncation. If false (default), generate benchmark matmuls where results are truncated and cast to the input element type.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[],
        help="List of matmul variants to filter benchmarks by. Default to all variants: NN, NT, TN, and TT.",
    )
    parser.add_argument(
        "--tag_regex",
        help="Regular expression for allowed benchmark tags. Defaults to all tags allowed.",
        default=".*",
    )
    parser.add_argument(
        "--config_regex",
        help="Regular expression for allowed benchmark configurations. Defaults to all allowed.",
        default=".*",
    )
    parser.add_argument(
        "--roofline",
        help="Comma separated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--backend",
        choices=["iree", "wave", "wavegqa", "torch"],
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
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=None,
        help="Maximum number of kernels to benchmark.",
    )

    args = parser.parse_args()
    # Handle default values here, since list args are not compatible with defaulted lists.
    requested_dtypes = ["f16"] if not args.dtypes else list(args.dtypes)
    requested_variants = (
        ["NN", "NT", "TN", "TT"] if not args.variants else list(args.variants)
    )

    logging.basicConfig(level=args.log_level)

    if args.roofline:
        for dtype in requested_dtypes:
            roofline(
                args.roofline,
                f"{args.plot.split('.')[0]}_{dtype}.png",
                args.batch,
                dtype,
                args.model,
            )
        sys.exit()

    backend_name = args.backend

    configs = []
    for dtype in requested_dtypes:
        configs += get_gemm_configs(dtype, backend_name, args.raw_accumulators)
    configs = get_matching_configs(
        configs,
        requested_variants,
        args.tag_regex,
        args.config_regex,
    )
    configs = reduce_configs(configs, args.max_kernels)
    print(f"Generated {len(configs)} gemm configs.")

    output_csv_base = f"gemm_{backend_name}"
    if args.raw_accumulators:
        output_csv_base += "_raw_accumulators"
    output_csv_path = Path(f"results/gemm/{output_csv_base}.csv")

    if backend_name not in ["iree", "wave", "wavegqa"]:
        benchmark_extern_gemm_kernels(backend_name, configs, output_csv_path)
        exit(0)

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "gemm" / backend_name / "mlir"
    vmfb_dir = repo_root / "gemm" / backend_name / "vmfb"
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    extra_compiler_args = ["--" + x for x in list(args.Xiree_compile)]
    device = "local-task" if args.target == "host_cpu" else args.device

    vmfb_dict = compile_gemm_kernels(
        backend_name,
        configs,
        kernel_dir,
        vmfb_dir,
        dump_dir,
        target,
        extra_compiler_args,
    )

    benchmark_gemm_kernels(backend_name, vmfb_dict, output_csv_path, args.iterations)
