import traceback
from dataclasses import dataclass, field
import pandas as pd
import json
from os import PathLike
from typing import List, Optional
from kernel_bench.config.base import OpConfig
from kernel_bench.tuning.hyperparam.paradigm.tree import MultiPassTreeTuner
from kernel_bench.tuning.hyperparam.parallel_tuning import ParallelTuner
from kernel_bench.utils.print_utils import get_logger
from kernel_bench.utils.paths import PathConfig
from ..utils.bench_utils import (
    BenchmarkResult,
    write_results_to_csv,
    reduce_configs,
    ConfigList,
    write_results_to_json,
)
from .template import KernelBenchmark, KernelValidationError, batch_benchmark
from kernel_bench.core.base import create_benchmark


@dataclass
class BenchmarkRunner:
    backend: str
    kernel_type: str
    configs: ConfigList
    device: str
    path_config: PathConfig = field(default_factory=PathConfig.default)

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
            tuned_data = json.load(file)

        improved_configs = {
            name: data
            for name, data in tuned_data.items()
            if data.get("improvement", False) and data.get("speedup", 0) > 1
        }

        speedups = [tune_result["speedup"] for tune_result in improved_configs.values()]
        avg_speedup = sum(speedups) / len(speedups)
        avg_speedup_percent = (avg_speedup - 1) * 100
        self.logger.info(
            f"Loading tuned config with average speedup of +{avg_speedup_percent:.2f}%"
        )

        # self.configs = [
        #     (tag, config)
        #     for tag, config in self.configs
        #     if config.get_name() in improved_configs.keys()
        # ]

        self.specs = {
            kernel_name: tune_result.get("hyperparams", {})
            for kernel_name, tune_result in improved_configs.items()
        }

    def save_results(self, results: List[BenchmarkResult]):
        if len(results) == 0:
            return None

        output_base = self.title or f"{self.kernel_type}_{self.backend}"

        output_csv_dir = self.path_config.csv_for(self.kernel_type)
        output_json_dir = self.path_config.json_for(self.kernel_type)

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
            "path_config": self.path_config,
        }
        try:
            bench = create_benchmark(
                self.kernel_type, self.backend, kwargs, serialize=False
            )
        except KernelValidationError:
            return None
        except Exception:
            self.logger.error(
                f"Failed to benchmark kernel {config.get_name()}\n{traceback.format_exc()}"
            )
            return None

        tuned_config = self.specs.get(config.get_name())
        if tuned_config:
            bench.load_tuned_config(tuned_config)

        return bench

    def _load_benches(self):
        """Create benchmark instances for all configurations."""
        benches = [self._create_benchmark(tag, config) for tag, config in self.configs]
        self._benches = [bench for bench in benches if bench]
        if len(self._benches) != len(self.configs):
            self.logger.info(
                f"Filtered {len(self._benches)} {self.kernel_type} configs for benchmarking on backend {self.backend}."
            )

        if self.max_kernels:
            self.reduce_benches(self.max_kernels)
            self.logger.info(
                f"Reduced to {len(self._benches)} {self.kernel_type} configs for benchmarking on backend {self.backend}."
            )

    def benchmark_kernels(self, validate_numerics=True) -> List[BenchmarkResult]:
        """
        Run benchmarks sequentially. Compiles all IREE-based kernels beforehand.
        """
        self._load_benches()

        results = batch_benchmark(
            self._benches,
            self.device,
            self.num_iterations,
            validate_numerics=validate_numerics,
            verbose=True,
        )

        self.save_results(results)

        return results

    def tune_kernels(
        self,
        num_trials: int = 100,
    ):
        """Runs benchmarks sequentially after parallel compilation."""
        self._load_benches()

        tuning_result_basename = f"{self.kernel_type}_{self.backend}_tuned_results.json"
        tuning_result_path = (
            self.path_config.tuning_for(self.kernel_type) / tuning_result_basename
        )

        tuning_paradigm = MultiPassTreeTuner()
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
