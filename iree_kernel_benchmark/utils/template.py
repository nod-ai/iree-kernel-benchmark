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
import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .tuning import TuningConstraint
from .bench_utils import bench_summary_process, run_iree_command, write_results_to_csv
from wave_lang.kernel.wave.constraints import MMAType
from .wave_utils import TuningSpec


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
        debug: bool = False,
    ) -> Tuple[float, bool]:
        exec_args = [
            "iree-benchmark-module",
            f"--device={self.device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            f"--benchmark_repetitions={num_iterations}",
            *config.get_runtime_args(self.backend),
        ]

        ret_value, cmd_out, cmd_err = run_iree_command(exec_args)
        benchmark_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_mean_time_us = benchmark_mean_time_ms * 1000
        ok = ret_value == 0 and benchmark_mean_time_us > 0

        return benchmark_mean_time_us, ok

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

        def to_mma(str_config: list[str]) -> tuple[MMAType]:
            return tuple([MMAType[mma_name] for mma_name in str_config])

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

            flops = config.get_flops()
            byte_count = config.get_byte_count()

            arithmetic_intensity = flops / byte_count
            if benchmark_mean_time_us == 0:
                tflops_per_second = 0
            else:
                tflops_per_second = (flops / 1e12) / (benchmark_mean_time_us / 1e6)

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
                config, vmfb_filename, self.num_iterations, self.debug
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
        tuning_results = {}
        tuning_dir = Path(f"results/tuning")
        tuning_result_basename = f"{self.kernel_type}_{self.backend}_tuned_results.json"
        tuning_result_path = tuning_dir / self.kernel_type / tuning_result_basename
        os.makedirs(os.path.dirname(tuning_result_path), exist_ok=True)

        def tune_config(config: Tuple[str, OpConfig]):
            tuning_result: tuple[
                float,
                TuningSpec,
                tuple[MMAType],
            ] = (
                math.inf,
                None,
                mfma_configs[0],
            )

            config_tag, kernel = config
            config_name = kernel.get_name()

            def objective(trial: optuna.Trial) -> float:
                nonlocal tuning_result

                block_sizes = [
                    trial.suggest_int(
                        constraint.name,
                        constraint.min,
                        constraint.max,
                        step=constraint.step,
                    )
                    for constraint in tiling_constraints
                ]

                tuning_spec_params = {
                    constraint.name: block_sizes[i]
                    for i, constraint in enumerate(tiling_constraints)
                }

                tuning_spec: TuningSpec = tuning_class(**tuning_spec_params)

                mfma_config = mfma_configs[
                    trial.suggest_categorical(
                        "MFMA_INDEX", list(range(len(mfma_configs)))
                    )
                ]

                local_kernel_dir = self.kernel_dir / self.kernel_type / self.backend

                mlir_dir = local_kernel_dir / "mlir"
                vmfb_dir = local_kernel_dir / "vmfb"
                os.makedirs(mlir_dir, exist_ok=True)
                os.makedirs(vmfb_dir, exist_ok=True)

                mlir_path = mlir_dir / f"{config_name}.mlir"
                vmfb_path = vmfb_dir / f"{config_name}.vmfb"

                try:
                    compile_success = self.compile_kernel(
                        kernel,
                        mlir_path,
                        vmfb_path,
                        mfma_variant=mfma_config,
                        spec=tuning_spec,
                    )
                    if not compile_success:
                        raise Exception(f"Compiling kernel {config_name} failed")
                except:
                    print("Failed to compile, skipping")
                    return math.inf

                try:
                    runtime, bench_success = self.bench_kernel(
                        kernel, vmfb_path, self.num_iterations, self.debug
                    )
                    if not bench_success:
                        raise Exception(f"Benchmarking kernel {config_name} failed")
                except:
                    print("Failed runtime, skipping")
                    return math.inf

                if runtime < tuning_result[0]:
                    tuning_result = (runtime, tuning_spec, mfma_config)

                return runtime

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

            best_runtime, best_spec, best_mfma = tuning_result
            print("Optimal spec", asdict(best_spec))

            if best_spec and best_mfma:
                tuning_results[config_name] = {
                    "block_sizes": asdict(best_spec),
                    "mfma_variant": [mfma.name for mfma in best_mfma],
                    "mean_microseconds": best_runtime,
                }

                with open(tuning_result_path) as file:
                    json.dump(tuning_results, file, indent=4)

        for config in self.configs:
            tune_config(config)
