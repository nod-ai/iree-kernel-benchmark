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

def generate_mlir_content(M, K, N, tA, tB, dtype):

    mlir_template_A = f"""
module {{
    func.func @main_0(%arg0: tensor<{K}x{M}x{dtype}>, %arg1: tensor<{K}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
        %cst = arith.constant 0.000000e+00 : {dtype}
        %0 = tensor.empty() : tensor<{M}x{N}x{dtype}>
        %1 = linalg.fill ins(%cst : {dtype}) outs(%0 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<{K}x{M}x{dtype}>, tensor<{K}x{N}x{dtype}>) outs(%1 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        return %2 : tensor<{M}x{N}x{dtype}>
    }}
}} 
"""

    mlir_template_B = f"""
module {{
    func.func @main_0(%arg0: tensor<{M}x{K}x{dtype}>, %arg1: tensor<{N}x{K}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
        %cst = arith.constant 0.000000e+00 : {dtype}
        %0 = tensor.empty() : tensor<{M}x{N}x{dtype}>
        %1 = linalg.fill ins(%cst : {dtype}) outs(%0 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<{M}x{K}x{dtype}>, tensor<{N}x{K}x{dtype}>) outs(%1 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        return %2 : tensor<{M}x{N}x{dtype}>
    }}
}} 
"""

    mlir_template = f"""module {{
    func.func @main_0(%arg0: tensor<{M}x{K}x{dtype}>, %arg1: tensor<{K}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
        %cst = arith.constant 0.000000e+00 : {dtype}
        %0 = tensor.empty() : tensor<{M}x{N}x{dtype}>
        %1 = linalg.fill ins(%cst : {dtype}) outs(%0 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        %2 = linalg.matmul ins(%arg0, %arg1 : tensor<{M}x{K}x{dtype}>, tensor<{K}x{N}x{dtype}>) outs(%1 : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
        return %2 : tensor<{M}x{N}x{dtype}>
    }}
}} 
"""
    if tA:
        return mlir_template_A
    if tB:
        return mlir_template_B
    return mlir_template


def compile_shape(tag, M, K, N, tA, tB, dtype, vmfb_dict):
    if tA == "T" and tB == "T":
        return f"Can't transpose both inputs"
    
    # Generate MLIR content
    mlir_content = generate_mlir_content(M, K, N, tA, tB, dtype)
    
    # Generate filenames
    filename = f"gemm/mlir/gemm_{M}_{K}_{N}_{dtype}"
    if tA == "T":
        filename += "_tA"
    elif tB == "T":
        filename += "_tB"
    mlir_filename = filename + ".mlir"
    filename = filename.replace("mlir", "vmfb")
    vmfb_filename = filename + ".vmfb"
    
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
    
    vmfb_dict[vmfb_filename] = [tag, M, K, N, tA, tB, dtype]
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
    
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline)
        sys.exit()
    
    shapes = []
    print(f"Generated {len(shapes)} gemm shapes.")
    
    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()
    all(shapes)
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

    vmfb_dir = repo_root / Path('gemm/vmfb')

    results = []
    index = 0
    output_csv = "results/iree_gemm.csv"

    for vmfb_filename, input_list in vmfb_dict.items():
        tag = input_list[0]
        vmfb_filename = vmfb_filename.split("/")[-1]
        name = vmfb_filename.split(".")[0]
        M = input_list[1]
        K = input_list[2]
        N = input_list[3]
        tA = input_list[4]
        tB = input_list[5]
        dtype = input_list[6]
        
        if tA == "T":
            inp1 = f"{K}x{M}x{dtype}"
            inp2 = f"{K}x{N}x{dtype}"
        elif tB == "T":
            inp1 = f"{M}x{K}x{dtype}"
            inp2 = f"{N}x{K}x{dtype}"
        else:
            inp1 = f"{M}x{K}x{dtype}"
            inp2 = f"{K}x{N}x{dtype}"

        exec_args = [
            "iree-benchmark-module",
            f"--device=hip",
            "--device_allocator=caching",
            f"--module={vmfb_dir}/{vmfb_filename}",
            "--function=main_0",
            f"--input={inp1}",
            f"--input={inp2}",
            "--benchmark_repetitions=3",
        ]

        # iree benchmark command for full sdxl pipeline
        ret_value, cmd_out = run_iree_command(exec_args)
        ok = ret_value == 0
        benchmark_gemm_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_gemm_mean_time_us = benchmark_gemm_mean_time_ms * 1000

        if "bf" in dtype:
            bytes_per_input = int(dtype[2:]) / 8
        else:
            bytes_per_input = int(dtype[1:]) / 8
        flops = 2 * M * N * K
        byte_count = bytes_per_input * (M * K + N * K + M * N)

        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_gemm_mean_time_us / 1e6)

        results.append((
            index, tag, name, M, N, K, dtype, tA, tB,
            round(benchmark_gemm_mean_time_us, 4),
            round(arithmetic_intensity, 4),
            round(tflops_per_second, 4),
            ok
        ))
        index += 1

    write_results_to_csv(results, output_csv)
    print(f"Results written to {output_csv}")
