from functools import total_ordering
import os
import logging
import subprocess
from pathlib import Path
import csv
from typing import Optional, Sequence
from collections import namedtuple, defaultdict
from dataclass_wizard import fromdict
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

from kernel_bench.utils.clustering import KernelConfigurationClustering
from kernel_bench.utils.print_utils import get_logger


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

    def get_dim_names(self) -> List[str]:
        return list(self.to_dict().keys())


@total_ordering
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

    def __eq__(self, other):
        if isinstance(other, BenchmarkResult):
            return self.mean_microseconds == other.mean_microseconds
        if isinstance(other, float):
            return self.mean_microseconds == other
        raise ValueError("Expected type BenchmarkResult")

    def __lt__(self, other):
        if isinstance(other, BenchmarkResult):
            return self.mean_microseconds < other.mean_microseconds
        if isinstance(other, float):
            return self.mean_microseconds < other
        raise ValueError("Expected type BenchmarkResult")


type ConfigList = List[Tuple[str, OpConfig]]


def generate_md5_hex(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def unit_to_microseconds(real_time: float, time_unit: str) -> float:
    unit_conversions = {
        "s": 1e6,
        "ms": 1e3,
        "us": 1,
        "ns": 1e-3,
    }
    assert time_unit in unit_conversions, f"Unsupported time unit: {time_unit}"
    return real_time * unit_conversions[time_unit]


def change_shape_dtype(shape: str, new_dtype: str):
    return "x".join(shape.split("x")[:-1] + [new_dtype])


def write_results_to_csv(results: list[BenchmarkResult], output_filename: os.PathLike):
    output_dir = os.path.dirname(Path(output_filename))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if len(results) == 0:
        get_logger().warn("No valid results")
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
        get_logger().warn("No valid results")
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

    get_logger().info(f"Roofline plot saved as '{out}'")


def reduce_configs(
    configs: List[Tuple[str, OpConfig]], max_kernels: int = None, seed: int = 42
) -> List[Tuple[str, OpConfig]]:
    if max_kernels is None or max_kernels >= len(configs):
        return configs

    def extract_features(g: OpConfig) -> tuple:
        return tuple(g.to_dict().values())

    def hash_features(feature_list: tuple) -> str:
        return "x".join(map(str, feature_list))

    # clustering
    config_shapes = [extract_features(g) for tag, g in configs]
    clusterer = KernelConfigurationClustering(
        scaling_method="minmax", clustering_method="kmeans", n_clusters=max_kernels
    )
    clusterer.fit(config_shapes)
    representatives = clusterer.get_representatives()
    representative_ids = {hash_features(shape) for shape in representatives}

    unique_mappings = {rep_id: None for rep_id in representative_ids}
    for tag, kernel in configs:
        kernel_id = hash_features(extract_features(kernel))
        if kernel_id in unique_mappings:
            unique_mappings[kernel_id] = (tag, kernel)

    configs = list(unique_mappings.values())

    return configs


def load_configs(
    config_path: os.PathLike, config_class: Type[OpConfig]
) -> List[Tuple[str, OpConfig]]:
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
    except:
        return []

    config_list = [
        (config["tag"], fromdict(config_class, config["problem"]))
        for config in config_data
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
