from functools import total_ordering
import os
from pathlib import Path
import csv
from typing import Optional
import sys
import json
from typing import Any, List, Tuple, Dict, Type
from contextlib import contextmanager
from dataclasses import dataclass
from dataclass_wizard import asdict

from kernel_bench.utils.clustering import KernelConfigurationClustering
from kernel_bench.utils.print_utils import get_logger
from kernel_bench.config.base import OpConfig


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


@contextmanager
def redirect_stderr_to_file(filepath):
    original_stderr = sys.stderr
    with open(filepath, "w") as f:
        sys.stderr = f
        try:
            yield
        finally:
            sys.stderr = original_stderr
