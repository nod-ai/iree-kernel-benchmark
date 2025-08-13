from collections import defaultdict
import math
import os
import random
import optuna
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import itertools
import pandas as pd
import time
import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wave_lang.kernel.wave.constraints import MMAType
import iree.runtime as ireert

from .tuning import TuningConstraint, tune_config_worker
from .bench_utils import (
    unit_to_microseconds,
    get_kernel_perf_stats,
    run_iree_command,
    write_results_to_csv,
    OpConfig,
    ConfigList,
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

    target: str = "gfx942"
    num_iterations: int = 3
    num_tuning_trials: int = 100

    _vmfb_dict: Dict[str, Tuple[str, OpConfig]] = field(default_factory=dict)

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
            # print(bench_result)
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

        # exec_args = [
        #     "iree-benchmark-module",
        #     f"--device={device or self.device}",
        #     "--device_allocator=caching",
        #     f"--module={vmfb_filename}",
        #     f"--benchmark_repetitions={num_iterations}",
        #     *config.get_runtime_args(self.backend),
        # ]

        # ret_value, cmd_out, cmd_err = run_iree_command(exec_args)
        # benchmark_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        # if not benchmark_mean_time_ms:
        #     return 0, False
        # benchmark_mean_time_us = benchmark_mean_time_ms * 1000
        # ok = ret_value == 0 and benchmark_mean_time_us > 0

        # return benchmark_mean_time_us, ok

    def _log(self, *args):
        if self.debug:
            print(*args)

    def reduce_configs(self, max_kernels: int = None, seed: int = 42):
        if max_kernels is None or max_kernels >= len(self.configs):
            return

        random.seed(seed)

        tag_to_configs = defaultdict(list)
        for tag, attrs in self.configs:
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
                            [
                                cfg
                                for cfg in configs_for_tag
                                if cfg not in selected_configs
                            ],
                            to_take,
                        )
                    )
                    remaining_budget -= to_take
                if remaining_for_tag > to_take:
                    next_round_overflow.append(tag)

            overflow_tags = next_round_overflow

        self.configs = selected_configs

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

    def save_results(
        self, selected_configs: ConfigList, runtimes_us: List[float], ok: List[bool]
    ):
        if len(selected_configs) == 0:
            return None

        output_csv_path = (
            f"results/{self.kernel_type}/{self.kernel_type}_{self.backend}.csv"
        )

        csv_dir = os.path.dirname(output_csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        index = 0
        results = []

        for i, (tag, config) in enumerate(selected_configs):
            benchmark_mean_time_us = runtimes_us[i]

            arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
                config, benchmark_mean_time_us
            )

            config_dict = config.to_dict()

            results.append(
                (
                    index,
                    tag,
                    config.get_name(),
                    *config_dict.values(),
                    round(benchmark_mean_time_us, 4),
                    round(arithmetic_intensity, 4),
                    round(tflops_per_second, 4),
                    ok[i],
                )
            )

            index += 1

        fieldnames = [
            "index",
            "tag",
            "name",
            *config_dict.keys(),
            "mean_microseconds",
            "arithmetic_intensity",
            "tflops",
            "ok",
        ]

        write_results_to_csv(results, output_csv_path, fieldnames)
        self._log(f"Results written to {output_csv_path}")

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

        runtimes = []
        statuses = []
        configs = []

        for vmfb_filename, value in bench_items:
            tag, config = value

            benchmark_mean_time_us, ok = self.bench_kernel(
                config,
                vmfb_filename,
                num_iterations=self.num_iterations,
                debug=self.debug,
            )

            runtimes.append(benchmark_mean_time_us)
            statuses.append(ok)
            configs.append((tag, config))

        return self.save_results(configs, runtimes, statuses)

    def benchmark_kernels_extern(self):
        bench_items = (
            tqdm(self.configs, desc="Benchmarking Attention Kernels")
            if self.debug
            else self.configs
        )

        runtimes = []
        statuses = []
        configs = []

        for tag, config in bench_items:
            benchmark_mean_time_us, ok = self.bench_kernel(
                config, "", self.num_iterations, self.debug
            )

            runtimes.append(benchmark_mean_time_us)
            statuses.append(ok)
            configs.append((tag, config))

        return self.save_results(configs, runtimes, statuses)

    def tune_kernels(
        self,
        mfma_configs: List[Tuple[MMAType]],
        tiling_constraints: List[TuningConstraint],
        tuning_class,
        num_trials: int = 100,
    ):
        tuning_dir = Path(f"results/tuning")
        tuning_result_basename = f"{self.kernel_type}_{self.backend}_tuned_results.json"
        tuning_result_path = tuning_dir / self.kernel_type / tuning_result_basename
        os.makedirs(os.path.dirname(tuning_result_path), exist_ok=True)

        # Create manager for shared resources
        manager = Manager()
        results_lock = manager.Lock()
        shared_results = manager.dict()

        # Prepare configs with device assignments
        worker_args = []
        total_configs = len(self.configs)
        num_gpus = min(8, total_configs)

        # Create progress manager
        progress_manager = ParallelProgressManager(total_configs, num_gpus)
        progress_manager.start_main_progress()

        for i, config in enumerate(self.configs):
            device_id = i % num_gpus
            worker_args.append(
                (
                    config,
                    device_id,
                    i + 1,
                    total_configs,
                    self.kernel_type,
                    self.backend,
                    self.kernel_dir,
                    self.num_iterations,
                    self.debug,
                    mfma_configs,
                    tiling_constraints,
                    tuning_class,
                    num_trials,
                    tuning_result_path,
                    results_lock,
                    shared_results,
                    self.compile_kernel,
                    self.bench_kernel,
                    get_kernel_perf_stats,
                    progress_manager.get_shared_state(),  # Add this
                    progress_manager.get_lock(),  # Add this
                )
            )

        # Run parallel tuning with progress monitoring
        import threading

        # Start a thread to refresh the display
        stop_refresh = threading.Event()

        def refresh_thread():
            while not stop_refresh.is_set():
                progress_manager.refresh_display()
                time.sleep(0.5)

        refresh_thread_obj = threading.Thread(target=refresh_thread)
        refresh_thread_obj.start()

        try:
            with Pool(processes=num_gpus) as pool:
                for result in pool.imap_unordered(tune_config_worker, worker_args):
                    if result[1] is not None:
                        pass
        finally:
            stop_refresh.set()
            refresh_thread_obj.join()
            progress_manager.close()

        # Store final results
        with open(tuning_result_path, "w") as file:
            json.dump(dict(shared_results), file, indent=4)
        print(f"\nSaved tuning results to {tuning_result_path}")
