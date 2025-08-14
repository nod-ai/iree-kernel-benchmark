from collections import defaultdict
import math
import os
import random
import optuna
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import itertools
import pandas as pd
import time
import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from wave_lang.kernel.wave.constraints import MMAType
import iree.runtime as ireert

from .tuning import TuningConstraint, tune_kernels_parallel
from .bench_utils import (
    BenchmarkResult,
    machine_to_hip_target,
    unit_to_microseconds,
    get_kernel_perf_stats,
    run_iree_command,
    write_results_to_csv,
    reduce_configs,
    OpConfig,
    ConfigList,
    write_results_to_json,
)
from .parallel import ParallelProgressManager
from .wave_utils import TuningSpec


@dataclass
class KernelBenchmark:
    backend: str
    kernel_type: str
    configs: ConfigList
    device: str

    kernel_dir: Path

    dump_dir: Optional[Path] = None
    debug: bool = False

    mfma_configs: Dict[str, Tuple[MMAType]] = field(default_factory=dict)
    specs: Dict[str, TuningSpec] = field(default_factory=dict)

    machine: str = "MI325X"
    target: str = field(init=False)
    num_iterations: int = 3

    _vmfb_dict: Dict[str, Tuple[str, OpConfig]] = field(default_factory=dict)

    def __post_init__(self):
        self.machine = self.machine.upper()
        self.target = machine_to_hip_target(self.machine)

    def compile_kernel(
        self,
        config: OpConfig,
        mlir_path: Path,
        vmfb_path: Path,
        extra_compiler_args: List[str] = [],
        mfma_variant: Optional[Tuple[MMAType]] = None,
        spec: Optional[TuningSpec] = None,
    ) -> bool:
        pass

    def bench_kernel(
        self,
        config: OpConfig,
        vmfb_filename: PathLike,
        num_iterations: int = 3,
        device: str = None,
        debug: bool = False,
    ) -> Tuple[float, bool]:

        extra_flags = {}
        func_name = None
        inputs = []
        for flag in config.get_runtime_args(self.backend):
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
                device=device or self.device,
                device_allocator="caching",
                benchmark_repetitions=num_iterations,
                **extra_flags,
            )
        except:
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

    def _log(self, *args):
        if self.debug:
            print(*args)

    def reduce_configs(self, max_kernels: int = None, seed: int = 42):
        self.configs = reduce_configs(self.configs, max_kernels, seed)

    def load_tuned_results(self, result_path: PathLike, spec_class_type):
        with open(result_path, "r") as file:
            tuned_data: dict[str, dict] = json.load(file)

        def to_mma(str_config: list[str]) -> tuple[MMAType] | MMAType:
            mma_types = tuple([MMAType[mma_name] for mma_name in str_config])
            if len(mma_types) == 1:
                return mma_types[0]
            return mma_types

        self.specs = {
            kernel_name: (spec_class_type(**tune_result["block_sizes"]))
            for kernel_name, tune_result in tuned_data.items()
        }
        self.mfma_configs = {
            kernel_name: to_mma(tune_result["mfma_variant"])
            for kernel_name, tune_result in tuned_data.items()
        }

    def save_results(self, results: List[BenchmarkResult]):
        if len(results) == 0:
            return None

        output_base = f"{self.kernel_type}_{self.backend}"

        output_csv_dir = Path(f"results/csv/{self.kernel_type}")
        output_json_dir = Path(f"results/json/{self.kernel_type}")

        output_csv_path = output_csv_dir / f"{output_base}.csv"
        output_json_path = output_json_dir / f"{output_base}.json"

        os.makedirs(output_csv_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)

        write_results_to_csv(results, output_csv_path)
        self._log(f"Results written to {output_csv_path}")

        write_results_to_json(results, output_json_path)
        self._log(f"Results written to {output_json_path}")

        return pd.read_csv(output_csv_path)

    def compile_kernels(self):
        vmfb_dict = {}

        local_kernel_dir = self.kernel_dir / self.kernel_type / self.backend
        mlir_dir = local_kernel_dir / "mlir"
        vmfb_dir = local_kernel_dir / "vmfb"

        os.makedirs(mlir_dir, exist_ok=True)
        os.makedirs(vmfb_dir, exist_ok=True)

        compilation_targets: List[Tuple] = [
            (
                tag,
                config,
                mlir_dir / f"{config.get_name()}.mlir",
                vmfb_dir / f"{config.get_name()}.vmfb",
            )
            for tag, config in self.configs
        ]

        def compile_args_generator():
            return itertools.starmap(
                lambda tag, config, mlir_path, vmfb_path: (
                    config,
                    mlir_path,
                    vmfb_path,
                    [],
                    self.mfma_configs.get(config.get_name()),
                    self.specs.get(config.get_name()),
                ),
                compilation_targets,
            )

        compilation_results: List[bool] = []

        if len(self.configs) < 5:
            compilation_results = [
                self.compile_kernel(*args) for args in compile_args_generator()
            ]
        else:
            num_cpus = max(1, cpu_count() - 20)
            self._log(f"Using {num_cpus} CPUs for parallel processing.")
            manager = Manager()
            shared_vmfb_dict = manager.dict()

            with Pool(num_cpus) as pool:
                compilation_results = list(
                    tqdm(
                        pool.istarmap(self.compile_kernel, compile_args_generator()),
                        total=len(self.configs),
                        desc="Compiling Attention Kernels",
                    )
                )
            vmfb_dict = shared_vmfb_dict

        error_count = 0
        for i, compilation_success in enumerate(compilation_results):
            if compilation_success:
                tag, config, mlir_path, vmfb_path = compilation_targets[i]
                vmfb_dict[str(vmfb_path)] = (tag, config)
            else:
                error_count += 1

        self._log(
            f"{len(self.configs) - error_count} Success, {error_count} Failed out of {len(self.configs)} configs"
        )
        self._log("Compilation process completed.")

        self._vmfb_dict = dict(vmfb_dict)

    def benchmark_kernels(self):
        bench_items = (
            tqdm(self._vmfb_dict.items(), desc="Benchmarking Attention Kernels")
            if self.debug
            else self._vmfb_dict.items()
        )

        results = []

        for index, (vmfb_filename, value) in enumerate(bench_items):
            tag, config = value

            benchmark_mean_time_us, ok = self.bench_kernel(
                config,
                vmfb_filename,
                num_iterations=self.num_iterations,
                debug=self.debug,
            )

            arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
                config, benchmark_mean_time_us
            )

            tuning_config = self.specs.get(config.get_name())
            if tuning_config:
                tuning_config = asdict(tuning_config)

            results.append(
                BenchmarkResult(
                    index=index,
                    machine=self.machine,
                    kernel_type=self.kernel_type,
                    backend=self.backend,
                    tag=tag,
                    name=config.get_name(),
                    dims=config.get_dim_names(),
                    shape=config.to_dict(),
                    problem=asdict(config),
                    tuning_config=tuning_config,
                    mean_microseconds=round(benchmark_mean_time_us, 4),
                    arithmetic_intensity=round(arithmetic_intensity, 4),
                    tflops=round(tflops_per_second, 4),
                    ok=ok,
                )
            )

        return self.save_results(results)

    def benchmark_kernels_extern(self):
        bench_items = (
            tqdm(self.configs, desc="Benchmarking Attention Kernels")
            if self.debug
            else self.configs
        )

        results = []

        for index, (tag, config) in enumerate(bench_items):
            benchmark_mean_time_us, ok = self.bench_kernel(
                config, "", self.num_iterations, self.debug
            )

            arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
                config, benchmark_mean_time_us
            )

            results.append(
                BenchmarkResult(
                    index=index,
                    machine=self.machine,
                    kernel_type=self.kernel_type,
                    backend=self.backend,
                    tag=tag,
                    name=config.get_name(),
                    dims=config.get_dim_names(),
                    shape=config.to_dict(),
                    problem=asdict(config),
                    tuning_config=None,
                    mean_microseconds=round(benchmark_mean_time_us, 4),
                    arithmetic_intensity=round(arithmetic_intensity, 4),
                    tflops=round(tflops_per_second, 4),
                    ok=ok,
                )
            )

        return self.save_results(results)

    def tune_kernels(
        self,
        mfma_configs: List[Tuple[MMAType]],
        tiling_constraints: List[TuningConstraint],
        tuning_class: Type[TuningSpec],
        num_trials: int = 100,
    ):
        tuning_dir = Path(f"results/tuning")
        tuning_result_basename = f"{self.kernel_type}_{self.backend}_tuned_results.json"
        tuning_result_path = tuning_dir / self.kernel_type / tuning_result_basename

        local_kernel_dir = self.kernel_dir / self.kernel_type / self.backend

        tune_kernels_parallel(
            self.configs,
            mfma_configs,
            tiling_constraints,
            tuning_class,
            self.compile_kernel,
            self.bench_kernel,
            local_kernel_dir,
            tuning_result_path,
            self.num_iterations,
            num_trials,
            self.debug,
            save_results=True,
        )

        print(f"Saved results to {tuning_result_path}")
