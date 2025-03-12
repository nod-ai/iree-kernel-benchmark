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
from .problems import get_gemm_configs, get_tk_gemm_configs, get_matching_configs


def compile_gemm(tag, config, kernel_dir, vmfb_dir, target, extra_compiler_args, tk, dump_dir=None):
    if dump_dir:
        name = config.get_name()
        dpath = os.path.join(dump_dir, name)
        extra_compiler_args.extend([
            f"--iree-hal-dump-executable-files-to={dpath}"
        ])
    mlir_file, vmfb_file = compile_gemm_config(config, kernel_dir, vmfb_dir, target, extra_compiler_args, tk)
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

    parser.add_argument("--target", help="The IREE hip target to compile for. The special value host_cpu results in a llvm-cpu benchmark instead of HIP, compiled for the host CPU.", type=str, default="gfx942")
    parser.add_argument("--device", help="The IREE device to execute benchmarks on", type=str, default="hip")
    parser.add_argument(
        "--Xiree_compile",
        nargs='+',
        default=[],
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`"
    )
    parser.add_argument(
        "--dtypes", nargs='+', default=[], help="List of data types to benchmark. Defaults to all supported types."
    )
    parser.add_argument(
        "--variants",
        nargs='+',
        default=[],
        help="List of matmul variants to benchmark. Default to all variants: NN, NT, TN, and TT."
    )
    parser.add_argument(
        "--tag_regex",
        help="Regular expression for allowed benchmark tags. Defaults to all tags allowed.",
        default=".*"
    )
    parser.add_argument("--roofline", help="Comma separated csv file list to generate roofline plot with", default=None)
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument("--batch", help="roofline on certain batch", type=int, default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--tk",
        action="store_true",
        default=False,
        help="Run gemm kernels using Turbine Kernels",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="Directory to which executable files will be dumped."
    )
    parser.add_argument(
        "--raw_accumulators",
        action='store_true',
        help="If true, benchmark matmuls returning the raw accumulator type with no truncation. If false (default), the results are truncated and cast to the input element type."
    )

    args = parser.parse_args()
    # Handle default values here, since list args are not compatible with defaulted lists.
    requested_dtypes = ["f16", "bf16", "i8"] if not args.dtypes else list(args.dtypes)
    requested_variants = ["NN", "NT", "TN", "TT"] if not args.variants else list(args.variants)

    logging.basicConfig(level=args.log_level)

    if args.roofline:
        for dtype in requested_dtypes:
            roofline(args.roofline, f"{args.plot.split('.')[0]}_{dtype}.png", args.batch, dtype, args.model)
        sys.exit()

    tk = args.tk
    configs = get_tk_gemm_configs() if tk else get_gemm_configs()
    configs = get_matching_configs(configs, requested_dtypes, requested_variants, args.tag_regex, args.raw_accumulators)
    print(f"Generated {len(configs)} gemm configs.")

    num_cpus = max(1, max(cpu_count() // 2, 1))
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "gemm" / "mlir"
    vmfb_dir = repo_root / "gemm" / "vmfb"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    extra_compiler_args = ['--' + x for x in list(args.Xiree_compile)]
    dump_dir = args.dump_dir
    device = "local-task" if args.target == "host_cpu" else args.device

    compile_args = itertools.starmap(
        lambda tag, config: (tag, config, kernel_dir, vmfb_dir, target, extra_compiler_args, tk, dump_dir), configs
    )
    with Pool(num_cpus) as pool:
        compilation_results = list(tqdm(pool.starmap(compile_gemm, list(compile_args))))

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
    output_csv_base = "iree_gemm"
    if args.raw_accumulators:
        output_csv_base += "_raw_accumulators"
    if tk:
        output_csv_base += "_tk"
    output_csv = f"results/{output_csv_base}.csv"
    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for vmfb_filename, value in vmfb_dict.items():
        tag, config = value
        vmfb_hash = generate_md5_hex(vmfb_filename)
        name = config.get_name()

        inp1 = config.get_inp1()
        inp2 = config.get_inp2()

        exec_args = [
            "iree-benchmark-module",
            f"--device={device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            "--benchmark_repetitions=3",
            f"--input={inp1}",
            f"--input={inp2}",
        ]

        if tk:
            out_shape = config.get_out()
            exec_args.append(f"--input={out_shape}")
            exec_args += ["--function=isolated_benchmark"]
        else:
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

        results.append((
            index, tag, name, vmfb_hash, config.M, config.N, config.K, config.operand_element_type, config.tA, config.tB,
            round(benchmark_gemm_mean_time_us, 4),
            round(arithmetic_intensity, 4),
            round(tflops_per_second, 4),
            ok
        ))
        index += 1

    fieldnames = [
        'index',
        'tag',
        'name',
        'vmfb_hash',
        'M',
        'N',
        'K',
        'dtype',
        'tA',
        'tB',
        'mean_microseconds',
        'arithmetic_intensity',
        'tflops',
        'ok'
    ]

    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")
