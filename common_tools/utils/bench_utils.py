import os
import logging
import subprocess
from pathlib import Path
import csv
from typing import Sequence
from collections import namedtuple
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import sys
import hashlib

def generate_md5_hex(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    (
        stdout_v,
        stderr_v,
    ) = (
        proc.stdout,
        proc.stderr,
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout, proc.stderr
    logging.getLogger().error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stdout, proc.stderr

def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results

def bench_summary_process(ret_value, output):
    if ret_value == 1:
        # Output should have already been logged earlier.
        logging.getLogger().error("Running convolution benchmark failed. Exiting.")
        return

    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)
    benchmark_mean_time = float(benchmark_results[3].time.split()[0])
    
    return benchmark_mean_time

def write_results_to_csv(results : list[tuple] | list[list] | list[dict], output_filename: str, fieldnames: []):
    if len(results) == 0:
        print('No valid results')
        return
    
    fieldnames = fieldnames

    with open(output_filename, 'w', newline='') as f:
        if isinstance(results[0], list) or isinstance(results[0], tuple):
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        elif isinstance(results[0], dict):
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
        else:
            print('Invalid result format')
            return
        
        for result in results:
            writer.writerow(result)

def filter_batch(data, b):
    data_new = []
    for row in data:
        if ("B" in row and int(row["B"]) == b):
            data_new.append(row)
        elif ("N" in row and int(row["N"]) == b):
            data_new.append(row)
    return data_new


def filter_dtype(data, dtype):
    data_new = []
    for row in data:
        if ("input_dtype" in row and row["input_dtype"] == dtype) or ("dtype" in row and row["dtype"] == dtype):
            data_new.append(row)
    return data_new

def filter_model(data, model):
    data_new = []
    for row in data:
        if ("tag" in row and model in row["tag"]):
            data_new.append(row)
    return data_new

def roofline(results=None, out=None, batch=None, dtype=None, model=None, **kwargs):
    """Generate a roofline plot of GEMM performance from multiple result files and save raw data as CSV."""
    if results is None:
        raise ValueError("No result files provided")
    if out is None:
        raise ValueError("No output file path provided")

    files = results.split(',')
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    
    plt.figure(figsize=(12, 8))

    for idx, result_file in enumerate(files):
        data = []
        with open(result_file.strip(), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = {k: float(v) if k in ['index', 'mean_microseconds', 'arithmetic_intensity', 'tflops'] else v for k, v in row.items()}
                row['ok'] = True if 'ok' not in row else row['ok'] == 'True'
                data.append(row)
        if batch:
            data = filter_batch(data, batch)
        if dtype:
            data = filter_dtype(data, dtype)
        if model:
            data = filter_model(data, model)
        if len(data) == 0:
            raise ValueError("No data to plot. If you set filters, there were no kernels with the target config")
        x = [item['arithmetic_intensity'] for item in data]
        y = [item['tflops'] for item in data]
        
        plt.scatter(x, y, alpha=0.6, color=next(colors), label=result_file.strip())
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (TFLOP/s)')
    plt.title('Roofline Plot of Kernel Performance')

    tflops_map = {
        "f32": 653.7,
        "f16": 1307.4,
        "bf16": 1307.4,
        "f8E4M3FNUZ": 2614.9,
        "i8": 2614.9,
    }
    
    peak_memory_bandwidth = 5.3
    if dtype is not None:
        peak_compute = tflops_map[dtype]
    else:
        peak_compute = 1307.4

    x_range = np.logspace(np.log10(min(x)), np.log10(max(max(x), 150)), 100)
    y_memory = peak_memory_bandwidth * x_range
    y_compute = np.full_like(x_range, peak_compute)
    
    y_cutoff = 1300
    mask = (peak_memory_bandwidth * x_range) <= y_cutoff
    x_filtered = x_range[mask]
    y_memory_filtered = peak_memory_bandwidth * x_filtered

    plt.plot(x_filtered, y_memory_filtered, 'r-', label='Memory Bound')
    plt.plot(x_range, y_compute, 'g-', label='Compute Bound')
    plt.plot(x_range, np.minimum(y_memory, y_compute), 'k-', linewidth=2, label='Roofline')
    
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.text(x_range[-1], peak_compute, f'{peak_compute:.1f} TFLOP/s', 
             verticalalignment='bottom', horizontalalignment='right')

    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Roofline plot saved as '{out}'")
