import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
from pathlib import Path
import csv
import argparse
import sys
from utils import *
from problems import *

def generate_mlir_content(B, H, S_Q, S_KV, DH, dtype):
    key_shape = f"[{B},{H},{S_KV},{DH}]"
    query_shape = f"[{B},{H},{S_Q},{DH}]"
    value_shape = f"[{B},{H},{S_KV},{DH}]"
    output_shape = f"[{B},{H},{S_Q},{DH}]"
    mlir_dtype = dtype
    mlir_template = f"""
module {{
    func.func @main_0(%295 : !torch.vtensor<{query_shape},{mlir_dtype}>, %298 : !torch.vtensor<{key_shape},{mlir_dtype}>, %301 : !torch.vtensor<{value_shape},{mlir_dtype}>) -> !torch.vtensor<{output_shape},{mlir_dtype}> {{
        %false_371 = torch.constant.bool false
        %float0.000000e00 = torch.constant.float 0.000000e+00
        %none_372 = torch.constant.none
        %none_373 = torch.constant.none
        %282:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%295, %298, %301, %float0.000000e00, %false_371, %none_372, %none_373) : (!torch.vtensor<{query_shape},{mlir_dtype}>, !torch.vtensor<{key_shape},{mlir_dtype}>, !torch.vtensor<{value_shape},{mlir_dtype}>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<{output_shape},{mlir_dtype}>, !torch.vtensor<[{B},{H},{S_Q}], f32>)
        return %282#0 : !torch.vtensor<{output_shape},{mlir_dtype}>
    }}
}} 
"""
    return mlir_template


def compile_shape(tag, B, H, S_Q, S_KV, DH, dtype, vmfb_dict):
    # Generate MLIR content
    mlir_content = generate_mlir_content(B, H, S_Q, S_KV, DH, dtype)
    
    # Generate filenames
    mlir_filename = f"attention/mlir/attention_B{B}_H{H}_SQ{S_Q}_SKV{S_KV}_DH{DH}_{dtype}.mlir"
    vmfb_filename = f"attention/vmfb/attention_B{B}_H{H}_SQ{S_Q}_SKV{S_KV}_DH{DH}_{dtype}.vmfb"
    
    # Write MLIR content to file
    with open(mlir_filename, 'w') as f:
        f.write(mlir_content)
    
    # Compile MLIR to VMFB
    exec_args = [
        "iree-compile",
        f"{mlir_filename}",
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
        "-o",
        f"{vmfb_filename}",
    ]
    ret_value, stdout = run_iree_command(exec_args)
    
    vmfb_dict[vmfb_filename] = [tag, B, H, S_Q, S_KV, DH, dtype]
    if ret_value == 0:
        return f"Successfully compiled {mlir_filename} to {vmfb_filename}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level",
    )
    parser.add_argument("--roofline", help="Comma seperated csv file list to generate roofline plot with", default=None)
    parser.add_argument("--plot", help="location to save plot", default=None)
    
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline, args.plot)
        sys.exit()
    
    shapes = []
    print(f"Generated {len(shapes)} attention shapes.")
    
    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()
    flash_attention(shapes)
    shape_idx = 0
    for shape in shapes:
        shape += (vmfb_dict,)
        shapes[shape_idx] = shape
        shape_idx += 1

    with Pool(num_cpus) as pool:
        results = list(tqdm(pool.starmap(compile_shape, shapes)))
    
    error_count = 0
    for result in results:
        if 'error' in result.lower():
            # print(result)
            error_count += 1
    print(f'{len(shapes) - error_count} Success, {error_count} Failed out of {len(shapes)} shapes')

    print("Compilation process completed.")

    repo_root = Path(__file__).parent.parent

    vmfb_dir = repo_root / Path('attention/vmfb')

    results = []
    index = 0
    output_csv = "results/iree_attention.csv"

    for vmfb_filename, input_list in vmfb_dict.items():
        tag = input_list[0]
        vmfb_filename = vmfb_filename.split("/")[-1]
        name = vmfb_filename.split(".")[0]
        B = input_list[1]
        H = input_list[2]
        S_Q = input_list[3]
        S_KV = input_list[4]
        DH = input_list[5]
        dtype = input_list[6]

        query_shape = f"{B}x{H}x{S_Q}x{DH}x{dtype}"
        key_shape = f"{B}x{H}x{S_KV}x{DH}x{dtype}"
        value_shape = f"{B}x{H}x{S_KV}x{DH}x{dtype}"

        exec_args = [
            "iree-benchmark-module",
            f"--device=hip",
            "--device_allocator=caching",
            f"--module={vmfb_dir}/{vmfb_filename}",
            "--function=main_0",
            f"--input={query_shape}",
            f"--input={key_shape}",
            f"--input={value_shape}",
            "--benchmark_repetitions=3",
        ]

        # iree benchmark kernels
        ret_value, cmd_out = run_iree_command(exec_args)
        ok = ret_value == 0
        benchmark_gemm_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_gemm_mean_time_us = benchmark_gemm_mean_time_ms * 1000

        if "bf" in dtype:
            bytes_per_input = int(dtype[2:]) / 8
        else:
            bytes_per_input = int(dtype[1:]) / 8
        flops = 4 * S_Q * S_KV * DH * B * H
        byte_count = bytes_per_input * B * H * (2 * S_KV * DH + 2 * S_Q * DH + S_Q * S_KV)

        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_gemm_mean_time_us / 1e6)

        results.append((
            index, tag, name, B, H, S_Q, S_KV, DH, dtype,
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
        'B', 
        'H', 
        'S_Q', 
        'S_KV', 
        'DH',
        'dtype',
        'mean_microseconds',
        'arithmetic_intensity',
        'tflops',
        'ok'
    ]

    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")
