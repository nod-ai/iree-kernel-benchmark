import os
import logging
import subprocess
from pathlib import Path
import csv
from typing import Optional, Sequence
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
from typing import Any, List, Tuple, Dict, Type
from contextlib import contextmanager
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import iree.runtime as ireert

import sympy
import wave_lang.kernel.lang as tkl
from wave_lang.kernel._support.indexing import index_symbol


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

    @abstractmethod
    def get_shared_mem_bytes(self, spec: Dict[str, Any]) -> int:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_dim_names(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class BenchmarkResult:
    machine: str
    kernel_type: str
    backend: str
    tag: str
    name: str
    dims: List[str]
    shape: Dict[str, Any]
    problem: Dict[str, Any]
    tuning_config: Optional[Dict[str, Any]]
    mean_microseconds: float
    arithmetic_intensity: float
    tflops: float
    ok: bool


type ConfigList = List[Tuple[str, OpConfig]]

HIP_TARGETS = {
    "mi100": "gfx908",
    "mi210": "gfx90a",
    "mi250": "gfx90a",
    "mi300a": "gfx942",
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi350x": "gfx950",
    "mi355x": "gfx950",
    "v710": "gfx1101",
    "w7700": "gfx1101",
    "w7800": "gfx1100",
    "w7900": "gfx1100",
    "rx7700xt": "gfx1101",
    "rx7800xt": "gfx1101",
    "rx7900xt": "gfx1100",
    "rx7900xtx": "gfx1100",
    "rx9060xt": "gfx1200",
    "rx9070": "gfx1201",
    "rx9070xt": "gfx1201",
    "r9700": "gfx1201",
}


def machine_to_hip_target(machine_name: str):
    target = HIP_TARGETS.get(machine_name.lower())
    if target is None:
        print(f"Could not find valid hip target for machine {machine_name}")
    return target


def generate_md5_hex(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


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


def unit_to_microseconds(real_time: float, time_unit: str) -> float:
    unit_conversions = {
        "s": 1e6,
        "ms": 1e3,
        "us": 1,
        "ns": 1e-3,
    }
    assert time_unit in unit_conversions, f"Unsupported time unit: {time_unit}"
    return real_time * unit_conversions[time_unit]


def write_results_to_csv(results: list[BenchmarkResult], output_filename: os.PathLike):
    output_dir = os.path.dirname(Path(output_filename))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if len(results) == 0:
        print("No valid results")
        return

    fieldnames = [
        "index",
        "machine",
        "kernel_type",
        "backend",
        "tag",
        "name",
        *list(results[0].shape.keys()),
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "ok",
    ]

    with open(output_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "tag": result.tag,
                    "name": result.name,
                    "machine": result.machine,
                    "kernel_type": result.kernel_type,
                    "backend": result.backend,
                    **result.shape,
                    "mean_microseconds": result.mean_microseconds,
                    "arithmetic_intensity": result.arithmetic_intensity,
                    "tflops": result.tflops,
                    "ok": result.ok,
                }
            )


def write_to_json_file(data: Any, output_filename: os.PathLike, indent=4):
    output_dir = os.path.dirname(Path(output_filename))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, "w") as file:
        json.dump(data, file, indent=indent)


def write_results_to_json(results: list[BenchmarkResult], output_filename: os.PathLike):
    if len(results) == 0:
        print("No valid results")
        return

    results_json = [asdict(result) for result in results]

    write_to_json_file(results_json, output_filename)


def get_kernel_perf_stats(
    config: OpConfig, benchmark_mean_time_us: float
) -> Tuple[float, float]:
    flops = config.get_flops()
    byte_count = config.get_byte_count()

    arithmetic_intensity = flops / byte_count
    if benchmark_mean_time_us <= 0:
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
) -> List[Tuple[str, OpConfig]]:
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


def bench_kernel_ireert(
    vmfb_filename: os.PathLike,
    iree_args: List[str],
    num_iterations: int = 3,
    device: str = None,
) -> Tuple[float, bool]:

    # print(
    #     f'iree-benchmark-module --device={device} --module={vmfb_filename} {" ".join(iree_args)}'
    # )

    extra_flags = {}
    func_name = None
    inputs = []
    for flag in iree_args:
        split_key_value = flag[2:].split("=")
        key = split_key_value[0]
        value = "=".join(split_key_value[1:])
        if key == "function":
            func_name = value
            continue
        if key == "input":
            inputs.append(value)
            continue
        extra_flags[key] = value

    try:
        bench_results = ireert.benchmark.benchmark_module(
            vmfb_filename,
            entry_function=func_name,
            inputs=inputs,
            device=device,
            device_allocator="caching",
            benchmark_repetitions=num_iterations,
            **extra_flags,
        )
    except Exception as e:
        print(e)
        return 0, False

    times = []
    for bench_result in bench_results:
        bench_name = bench_result.benchmark_name
        if bench_name.split("/")[-1] == "real_time":
            time_and_unit = bench_result.time.split(" ")
            assert (
                len(time_and_unit) == 2
            ), "expected the benchmark time to be the time and unit separated by a space."
            time_us = unit_to_microseconds(
                real_time=float(time_and_unit[0]),
                time_unit=time_and_unit[1],
            )
            times.append(time_us)

    if len(times) == 0:
        return 0, False

    benchmark_mean_time_us = sum(times) / float(len(times))
    return benchmark_mean_time_us, True


@contextmanager
def redirect_stderr_to_file(filepath):
    original_stderr = sys.stderr
    with open(filepath, "w") as f:
        sys.stderr = f
        try:
            yield
        finally:
            sys.stderr = original_stderr
