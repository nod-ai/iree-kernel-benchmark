import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import csv
import argparse
import sys
from utils import *
from conv_utils import *
from problems import get_conv_configs, get_conv_test_configs

from wave_conv_utils import compile_wave_conv_config

def compile_conv_iree(tag, config, kernel_dir, vmfb_dir, extra_compiler_args):
    mlir_file, vmfb_file, dump_path = compile_conv_config(config, kernel_dir, vmfb_dir, extra_compiler_args)
    return (tag, config, mlir_file, vmfb_file, dump_path)

def compile_conv_wave(tag, config, kernel_dir, vmfb_dir, extra_compiler_args):
    mlir_file, vmfb_file, dump_path = compile_wave_conv_config(config, kernel_dir, vmfb_dir, extra_compiler_args)
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
    parser.add_argument("--device", help="The IREE device to execute benchmarks on", type=str, default="hip")
    parser.add_argument(
        "--Xiree_compile",
        nargs='+',
        default=[],
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`."
    )
    parser.add_argument(
        "--roofline",
        help="Comma seperated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument("--batch", help="roofline on certain batch", type=int, default=None)
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument('--tk', help="Run conv kernels using Turbine Kernels", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline, args.plot, args.batch, args.dtype, args.model)
        sys.exit()

    # configs = get_conv_test_configs()
    configs = get_conv_configs()
    print(f"Generated {len(configs)} conv configs.")

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

    extra_compiler_args = ['--' + x for x in list(args.Xiree_compile)]
    compile_args = itertools.starmap(
        lambda tag, config: (tag, config, kernel_dir, vmfb_dir, extra_compiler_args), configs
    )
    compile_conv = compile_conv_wave if args.tk else compile_conv_iree
    with Pool(num_cpus) as pool:
        compilation_results = list(tqdm(pool.starmap(compile_conv, list(compile_args))))

    error_count = 0
    for tag, config, mlir_file, vmfb_file, dump_path in compilation_results:
        if vmfb_file:
            vmfb_dict[vmfb_file] = (tag, config, dump_path)
        else:
            error_count += 1
    print(
        f"{len(configs) - error_count} Success, {error_count} Failed out of {len(configs)} configs"
    )

    print("Compilation process completed.")

    results = []
    index = 0
    output_csv = "results/iree_conv_tk.csv" if args.tk else "results/iree_conv.csv"
    entrypoint = "isolated_benchmark" if args.tk else "main"
    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for vmfb_filename, value in vmfb_dict.items():
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
            "--benchmark_repetitions=3",
            f"--input={image_shape}",
            f"--input={filter_shape}",
        ]

        if args.tk:
            out_shape = config.get_out_shape()
            exec_args.append(f"--input={out_shape}")

        print(f"Running {vmfb_filename}...")
        # iree benchmark kernels
        ret_value, cmd_out, cmd_stderr = run_iree_command(exec_args)
        ok = ret_value == 0
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
