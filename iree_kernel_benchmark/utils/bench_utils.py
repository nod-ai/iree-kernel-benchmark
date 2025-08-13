import os
import logging
import subprocess
from pathlib import Path
import csv
from typing import Sequence
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import sys
import hashlib
import warnings
import random
import torch
import json
from typing import Any, List, Tuple, Dict
from contextlib import contextmanager
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


@dataclass
class OpConfig(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_flops(self) -> int:
        pass

    @abstractmethod
    def get_byte_count(self) -> int:
        pass

    @abstractmethod
    def get_runtime_args(self, backend_name: str) -> List[str]:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


type ConfigList = List[Tuple[str, OpConfig]]


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


def unit_to_microseconds(real_time: float, time_unit: str) -> float:
    unit_conversions = {
        "s": 1e6,
        "ms": 1e3,
        "us": 1,
        "ns": 1e-3,
    }
    assert time_unit in unit_conversions, f"Unsupported time unit: {time_unit}"
    return real_time * unit_conversions[time_unit]


def bench_summary_process(ret_value, output: bytes):
    if ret_value == 1:
        # Output should have already been logged earlier.
        logging.getLogger().error("Running benchmark failed. Exiting.")
        return

    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)

    for result in benchmark_results:
        if "real_time_mean" in result.benchmark_name:
            return float(result.time.split()[0])

    return float(benchmark_results[0].time.split()[0])


def write_results_to_csv(
    results: list[tuple] | list[list] | list[dict], output_filename: str, fieldnames: []
):
    if len(results) == 0:
        print("No valid results")
        return

    fieldnames = fieldnames

    with open(output_filename, "w", newline="") as f:
        if isinstance(results[0], list) or isinstance(results[0], tuple):
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        elif isinstance(results[0], dict):
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
        else:
            print("Invalid result format")
            return

        for result in results:
            writer.writerow(result)


def get_kernel_perf_stats(
    config: OpConfig, benchmark_mean_time_us: float
) -> Tuple[float, float]:
    flops = config.get_flops()
    byte_count = config.get_byte_count()

    arithmetic_intensity = flops / byte_count
    if benchmark_mean_time_us == 0:
        tflops_per_second = 0
    else:
        tflops_per_second = (flops / 1e12) / (benchmark_mean_time_us / 1e6)

    return arithmetic_intensity, tflops_per_second


def filter_batch(data, b):
    data_new = []
    for row in data:
        if "B" in row and int(row["B"]) == b:
            data_new.append(row)
        elif "N" in row and int(row["N"]) == b:
            data_new.append(row)
    return data_new


def filter_dtype(data, dtype):
    data_new = []
    for row in data:
        if ("input_dtype" in row and row["input_dtype"] == dtype) or (
            "dtype" in row and row["dtype"] == dtype
        ):
            data_new.append(row)
    return data_new


def filter_model(data, model):
    data_new = []
    for row in data:
        if "tag" in row and model in row["tag"]:
            data_new.append(row)
    return data_new


def roofline(results=None, out=None, batch=None, dtype=None, model=None, **kwargs):
    """Generate a roofline plot of GEMM performance from multiple result files and save raw data as CSV."""
    if results is None:
        raise ValueError("No result files provided")
    if out is None:
        raise ValueError("No output file path provided")

    files = results.split(",")
    colors = cycle(["b", "g", "r", "c", "m", "y", "k"])

    plt.figure(figsize=(12, 8))

    for idx, result_file in enumerate(files):
        data = []
        with open(result_file.strip(), mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                float_columns = [
                    "index",
                    "mean_microseconds",
                    "arithmetic_intensity",
                    "tflops",
                    "roofline_tflops",
                    "roofline_percent",
                ]
                row = {
                    k: (float(v) if k in float_columns else v) for k, v in row.items()
                }
                row["ok"] = True if "ok" not in row else row["ok"] == "True"
                data.append(row)
        if batch:
            data = filter_batch(data, batch)
        if dtype:
            data = filter_dtype(data, dtype)
        if model:
            data = filter_model(data, model)
        if len(data) == 0:
            warnings.warn(
                f"No data to plot with filters dtype={dtype}, batch={batch}, model={model} there were no kernels with the target config"
            )
            return
        x = [item["arithmetic_intensity"] for item in data]
        y = [item["tflops"] for item in data]

        plt.scatter(x, y, alpha=0.6, color=next(colors), label=result_file.strip())

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOP/byte)")
    plt.ylabel("Performance (TFLOP/s)")
    plt.title("Roofline Plot of Kernel Performance")

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

    plt.plot(x_filtered, y_memory_filtered, "r-", label="Memory Bound")
    plt.plot(x_range, y_compute, "g-", label="Compute Bound")
    plt.plot(
        x_range, np.minimum(y_memory, y_compute), "k-", linewidth=2, label="Roofline"
    )

    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.text(
        x_range[-1],
        peak_compute,
        f"{peak_compute:.1f} TFLOP/s",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Roofline plot saved as '{out}'")


def reduce_configs(
    configs: List[Tuple[str, Any]], max_kernels: int = None, seed: int = 42
) -> List[Tuple[str, Any]]:
    if max_kernels is None or max_kernels >= len(configs):
        return configs

    random.seed(seed)

    tag_to_configs = defaultdict(list)
    for tag, attrs in configs:
        tag_to_configs[tag].append((tag, attrs))

    tags = list(tag_to_configs.keys())
    total_tags = len(tags)

    target_per_tag = max_kernels // total_tags
    selected_configs = []
    remaining_budget = max_kernels
    overflow_tags = []

    for tag in tags:
        configs_for_tag = tag_to_configs[tag]
        if len(configs_for_tag) <= target_per_tag:
            selected_configs.extend(configs_for_tag)
            remaining_budget -= len(configs_for_tag)
        else:
            overflow_tags.append(tag)

    while remaining_budget > 0 and overflow_tags:
        per_tag_extra = max(1, remaining_budget // len(overflow_tags))
        next_round_overflow = []

        for tag in overflow_tags:
            configs_for_tag = tag_to_configs[tag]
            already_selected = sum(1 for c in selected_configs if c[0] == tag)
            remaining_for_tag = len(configs_for_tag) - already_selected
            to_take = min(per_tag_extra, remaining_for_tag)
            if to_take > 0:
                selected_configs.extend(
                    random.sample(
                        [cfg for cfg in configs_for_tag if cfg not in selected_configs],
                        to_take,
                    )
                )
                remaining_budget -= to_take
            if remaining_for_tag > to_take:
                next_round_overflow.append(tag)

        overflow_tags = next_round_overflow

    return selected_configs


def load_configs(
    config_path: os.PathLike, kernel_type: str, backend: str, config_class
) -> List[Tuple[str, Any]]:
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
    except:
        return []

    filtered_data = [
        config
        for config in config_data
        if config["kernelType"] == kernel_type and backend in config["allowedBackends"]
    ]

    config_list = [
        (str(config["tag"]), config_class(**config["problem"]))
        for config in filtered_data
    ]

    return config_list


@contextmanager
def redirect_stderr_to_file(filepath):
    original_stderr = sys.stderr
    with open(filepath, "w") as f:
        sys.stderr = f
        try:
            yield
        finally:
            sys.stderr = original_stderr
