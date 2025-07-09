import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import csv
import argparse
import sys
from ..utils import *
from .attention_utils import *
from .problems import get_attention_configs


def compile_attention(
    tag, config, kernel_dir, vmfb_dir, 
    extra_compiler_args=[], tk=False, dump_dir=None,
):
    if dump_dir:
        dump_dir = Path(dump_dir)
        dpath = dump_dir / config.get_name()
        extra_compiler_args.extend([f"--iree-hal-dump-executable-files-to={dpath}"])
    mlir_file, vmfb_file = compile_attention_config(config, kernel_dir, vmfb_dir, dump_dir, extra_compiler_args, tk)
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
        "--tk",
        action="store_true",
        default=False,
        help="Run attention kernels using Wave Kernels",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="Directory to which executable files will be dumped.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline, args.plot, args.batch, args.dtype, args.model)
        sys.exit()
    
    tk = args.tk

    configs = get_attention_configs()
    print(f"Generated {len(configs)} attention configs.")

    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()

    repo_root = Path(__file__).parent.parent
    backend_name = "wave" if tk else "iree"
    kernel_dir = repo_root / "attention" / backend_name / "mlir"
    vmfb_dir = repo_root / "attention" / backend_name / "vmfb"
    dump_dir = args.dump_dir
    device = args.device
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)

    compile_args = itertools.starmap(
        lambda tag, config: (tag, config, kernel_dir, vmfb_dir, [], tk, dump_dir), configs
    )
    with Pool(num_cpus) as pool:
        compilation_results = list(
            tqdm(pool.starmap(compile_attention, list(compile_args)))
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
    output_csv = "results/attention_wave.csv" if tk else "results/attention_iree.csv"
    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for vmfb_filename, value in tqdm(vmfb_dict.items(), desc="Benchmarking Attention Kernels"):
        tag, config = value
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
            "--benchmark_repetitions=3",
        ]

        if tk:
            out_shape : str = config.get_output_shape()
            out_shape = 'x'.join(out_shape.split('x')[:-1] + ['f32'])
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
