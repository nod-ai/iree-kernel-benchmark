import datetime
import os
import traceback
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from dataclasses import asdict, dataclass, field
import pandas as pd
import json
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Type
from wave_lang.kernel.wave.constraints import MMAType

from kernel_bench.tuning.hyperparam.paradigm.bayesian import BayesianTuningParadigm
from kernel_bench.tuning.hyperparam.paradigm.test_progress import ParallelProgressTester
from kernel_bench.tuning.hyperparam.paradigm.tree import MultiPassTreeTuner
from kernel_bench.tuning.hyperparam.parallel_tuning import ParallelTuner
from kernel_bench.tuning import tune_kernel_schedule
from kernel_bench.utils.print_utils import get_logger
from ..utils.bench_utils import (
    BenchmarkResult,
    get_kernel_perf_stats,
    write_results_to_csv,
    reduce_configs,
    OpConfig,
    ConfigList,
    write_results_to_json,
    write_to_json_file,
)
from .template import (
    KernelBenchmark,
    IREEKernelBenchmark,
    WaveKernelBenchmark,
    batch_benchmark,
    batch_compile_iree_benches,
)
from kernel_bench.core.base import BENCHMARKS, create_benchmark


@dataclass
class BenchmarkRunner:
    backend: str
    kernel_type: str
    configs: ConfigList
    device: str

    kernel_dir: Path

    dump_dir: Optional[Path] = None
    debug: bool = False

    machine: str = "MI325X"
    num_iterations: int = 3

    title: str = None
    max_kernels: Optional[int] = None

    def __post_init__(self):
        self._benches: List[KernelBenchmark] = None
        self.specs = {}
        self.machine = self.machine.upper()
        self.logger = get_logger()

    def reduce_benches(self, max_kernels: int = None, seed: int = 42):
        if max_kernels >= len(self._benches):
            return

        current_configs = [("null", bench.config) for bench in self._benches]
        selected_configs = {
            config.get_name()
            for tag, config in reduce_configs(current_configs, max_kernels, seed)
        }
        self._benches = [
            bench
            for bench in self._benches
            if bench.config.get_name() in selected_configs
        ]

    def load_tuned_results(self, result_path: PathLike):
        with open(result_path, "r") as file:
            tuned_data: dict[str, dict] = json.load(file)

        tuned_data = {
            kernel_name: tune_result
            for kernel_name, tune_result in tuned_data.items()
            if tune_result["improvement"]
        }

        speedups = [tune_result["speedup"] for tune_result in tuned_data.values()]
        avg_speedup = sum(speedups) / len(speedups)
        avg_speedup_percent = (avg_speedup - 1) * 100
        self.logger.info(
            f"Loading tuned config with average speedup of +{avg_speedup_percent:.2f}%"
        )

        self.configs = [
            (tag, config)
            for tag, config in self.configs
            if config.get_name() in tuned_data.keys()
        ]

        self.specs = {
            kernel_name: tune_result["hyperparams"]
            for kernel_name, tune_result in tuned_data.items()
            if tune_result["improvement"]
        }

    def save_results(self, results: List[BenchmarkResult]):
        if len(results) == 0:
            return None

        output_base = self.title or f"{self.kernel_type}_{self.backend}"

        output_csv_dir = Path(f"results/csv/{self.kernel_type}")
        output_json_dir = Path(f"results/json/{self.kernel_type}")

        output_csv_path = output_csv_dir / f"{output_base}.csv"
        output_json_path = output_json_dir / f"{output_base}.json"

        write_results_to_csv(results, output_csv_path)
        self.logger.info(f"Results written to {output_csv_path}")

        write_results_to_json(results, output_json_path)
        self.logger.info(f"Results written to {output_json_path}")

        return pd.read_csv(output_csv_path)

    def _create_benchmark(
        self, tag: str, config: OpConfig
    ) -> Optional[KernelBenchmark]:
        """Create appropriate benchmark instance based on backend type."""
        kwargs = {
            "tag": tag,
            "backend": self.backend,
            "kernel_type": self.kernel_type,
            "machine": self.machine,
            "config": config,
        }
        if "wave" in self.backend or "iree" in self.backend:
            kwargs.update(
                {
                    "kernel_dir": self.kernel_dir,
                    "dump_dir": self.dump_dir,
                }
            )

        try:
            bench = create_benchmark(self.kernel_type, self.backend, kwargs)
        except:
            return None

        tuned_config = self.specs.get(config.get_name())
        if tuned_config:
            bench.load_tuned_config(tuned_config)

        return bench

    def _load_benches(self):
        """Create benchmark instances for all configurations."""
        benches = [self._create_benchmark(tag, config) for tag, config in self.configs]
        self._benches = [bench for bench in benches if bench]

    def benchmark_kernels(self, validate_numerics=True) -> List[BenchmarkResult]:
        """
        Run benchmarks sequentially. Compiles all IREE-based kernels beforehand.
        """
        self._load_benches()
        if len(self._benches) != len(self.configs):
            self.logger.info(
                f"Filtered {len(self._benches)} {self.kernel_type} configs for benchmarking on backend {self.backend}."
            )

        if len(self._benches) == 0:
            return []

        if self.max_kernels:
            self.reduce_benches(self.max_kernels)
            self.logger.info(
                f"Reduced to {len(self._benches)} {self.kernel_type} configs for benchmarking on backend {self.backend}."
            )

        results = batch_benchmark(
            self._benches,
            self.device,
            self.num_iterations,
            validate_numerics=validate_numerics,
            verbose=True,
        )

        self.save_results(results)

        return results

    def tune_scheduling(self, max_iterations=100):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/tuning/{self.kernel_type}")
        output_path = (
            output_dir / f"{self.kernel_type}_{self.backend}_schedule_tuned.json"
        )

        results = {}

        for tag, config in tqdm(self.configs, desc="Tuning Scheduling"):
            self.logger.info(f"Tuning scheduling for kernel {config.get_name()}")
            result, runtime_us = tune_kernel_schedule(
                config,
                run_name=timestamp,
                load_kernel_func=self.load_kernel,
                compile_kernel_func=self.compile_kernel,
                bench_kernel_func=self.bench_kernel,
                kernel_dir=self.kernel_dir,
                max_iterations=max_iterations,
                extra_compile_options={},
            )
            arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
                config, runtime_us
            )
            results[config.get_name()] = {
                "result": result,
                "mean_microseconds": runtime_us,
                "arithmetic_intensity": arithmetic_intensity,
                "tflops": tflops_per_second,
                "problem": asdict(config),
            }
            write_to_json_file(results, output_path)

        return results

    def tune_kernels(
        self,
        num_trials: int = 100,
    ):
        """Runs benchmarks sequentially after parallel compilation."""
        self._load_benches()

        tuning_dir = Path(f"results/tuning")
        tuning_result_basename = f"{self.kernel_type}_{self.backend}_tuned_results.json"
        tuning_result_path = tuning_dir / self.kernel_type / tuning_result_basename

        # tuning_paradigm = BayesianTuningParadigm()
        tuning_paradigm = MultiPassTreeTuner()
        # tuning_paradigm = ParallelProgressTester()
        tuner = ParallelTuner(tuning_paradigm)
        tuner.tune_kernels(
            benches=self._benches,
            tuning_result_path=tuning_result_path,
            num_iterations=self.num_iterations,
            num_trials=num_trials,
            debug=self.debug,
            save_results=True,
        )

        self.logger.info(f"Saved results to {tuning_result_path}")
