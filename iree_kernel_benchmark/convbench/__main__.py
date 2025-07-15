import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import csv
import argparse
import sys
import re
from collections import OrderedDict
from ..utils import *
from .conv_utils import *
from .problems import get_conv_configs, get_tk_conv_configs, get_conv_test_configs

from .wave_conv_utils import compile_wave_conv_config


def compile_conv_iree(tag, config, kernel_dir, vmfb_dir, extra_compiler_args):
    mlir_file, vmfb_file, dump_path = compile_conv_config(
        tag, config, kernel_dir, vmfb_dir, extra_compiler_args
    )
    return (tag, config, mlir_file, vmfb_file, dump_path)


def compile_conv_wave(tag, config, kernel_dir, vmfb_dir, extra_compiler_args):
    mlir_file, vmfb_file, dump_path = compile_wave_conv_config(
        tag, config, kernel_dir, vmfb_dir, extra_compiler_args
    )
    return (tag, config, mlir_file, vmfb_file, dump_path)


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
        "--device",
        help="The IREE device to execute benchmarks on",
        type=str,
        default="hip",
    )
    parser.add_argument(
        "--Xiree_compile",
        nargs="+",
        default=[],
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`.",
    )
    parser.add_argument(
        "--filter-config",
        help="Only execute configs matching the provided regex",
        type=str,
    )
    parser.add_argument(
        "--roofline",
        help="Comma seperated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--tk",
        help="Run conv kernels using Turbine Kernels",
        action=argparse.BooleanOptionalAction,
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

    # configs = get_conv_test_configs()
    configs = get_tk_conv_configs() # if args.tk else get_conv_configs()
    print(f"Generated {len(configs)} conv configs.")

    configs = list(OrderedDict({config: None for config in configs}))
    print(f"Deduplicated to {len(configs)} conv configs.")

    if args.filter_config is not None:
        filter_regex = re.compile(args.filter_config)
        configs = list(
            filter(lambda config: filter_regex.match(config[1].get_name()), configs)
        )
        print(f"Filtered down to {len(configs)} conv configs.")

    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "conv" / "mlir"
    vmfb_dir = repo_root / "conv" / "vmfb"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    extra_compiler_args = ["--" + x for x in list(args.Xiree_compile)]
    compile_args = itertools.starmap(
        lambda tag, config: (tag, config, kernel_dir, vmfb_dir, extra_compiler_args),
        configs,
    )
    compile_conv = compile_conv_wave if args.tk else compile_conv_iree
    with Pool(num_cpus) as pool:
        compilation_results = list(
            tqdm(
                pool.istarmap(compile_conv, list(compile_args)),
                total=len(configs),
                desc="Compiling Conv Kernels"
            )
        )

    compile_error_count = 0
    for tag, config, mlir_file, vmfb_file, dump_path in compilation_results:
        if vmfb_file:
            vmfb_dict[vmfb_file] = (tag, config, dump_path)
        else:
            compile_error_count += 1
    print(
        f"{len(configs) - compile_error_count} Success, {compile_error_count} Failed out of {len(configs)} configs"
    )

    print("Compilation process completed.")

    results = []
    index = 0
    output_csv = "results/conv/conv_wave.csv" if args.tk else "results/conv/conv_iree.csv"
    entrypoint = "isolated_benchmark" if args.tk else "main"
    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    print(f'Results will be written to {Path(output_csv)}')

    run_error_count = 0
    for vmfb_filename, value in tqdm(vmfb_dict.items(), desc="Benchmarking Conv Kernels"):
        tag, config, dump_path = value
        name = config.get_name()

        image_shape = config.get_img_shape()
        filter_shape = config.get_kernel_shape()

        exec_args = [
            "iree-benchmark-module",
            f"--device={device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            f"--function={entrypoint}",
            f"--benchmark_repetitions={args.iterations}",
            f"--input={image_shape}",
            f"--input={filter_shape}",
        ]

        if args.tk:
            out_shape = config.get_out_shape()
            exec_args.append(f"--input={out_shape}")

        # iree benchmark kernels
        ret_value, cmd_out, cmd_stderr = run_iree_command(exec_args)
        ok = ret_value == 0
        if not ok:
            run_error_count += 1
        benchmark_conv_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_conv_mean_time_us = benchmark_conv_mean_time_ms * 1000

        flops = config.get_flops()
        byte_count = config.get_byte_count()

        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_conv_mean_time_us / 1e6)

        # Compute percentage of the roofline.
        # TODO: Make this target specific and move to common utils.
        tflops_map = {
            "f32": 653.7,
            "f16": 1307.4,
            "bf16": 1307.4,
            "f8E4M3FNUZ": 2614.9,
            "i8": 2614.9,
        }
        roofline_tflops = tflops_map[config.input_dtype]

        results.append(
            (
                index,
                tag,
                name,
                config.N,
                config.H,
                config.W,
                config.C,
                config.P,
                config.Q,
                config.F,
                config.S,
                config.input_dtype,
                config.output_dtype,
                round(benchmark_conv_mean_time_us, 4),
                round(arithmetic_intensity, 4),
                round(tflops_per_second, 4),
                roofline_tflops,
                round(tflops_per_second / roofline_tflops, 4),
                ok,
            )
        )
        index += 1

    fieldnames = [
        "index",
        "tag",
        "name",
        "B",
        "H",
        "W",
        "C",
        "P",
        "Q",
        "F",
        "S",
        "input_dtype",
        "output_dtype",
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "roofline_tflops",
        "roofline_percent",
        "ok",
    ]

    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")

    if compile_error_count != 0 or run_error_count != 0:
        exit(1)
    else:
        exit(0)
