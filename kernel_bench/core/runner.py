import datetime
import os
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
from kernel_bench.tuning.hyperparam.parallel_tuning import ParallelTuner
from kernel_bench.tuning import tune_kernel_schedule
from .utils import (
    BenchmarkResult,
    get_kernel_perf_stats,
    write_results_to_csv,
    reduce_configs,
    OpConfig,
    ConfigList,
    write_results_to_json,
    write_to_json_file,
)
from .template import KernelBenchmark, IREEKernelBenchmark, WaveKernelBenchmark
from kernel_bench.core.base import BENCHMARKS


@dataclass
class BenchmarkRunner:
    backend: str
    kernel_type: str
    configs: ConfigList
    device: str

    kernel_dir: Path

    dump_dir: Optional[Path] = None
    debug: bool = False

    hyperparams: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    machine: str = "MI325X"
    num_iterations: int = 3

    title: str = None

    _benches: List[KernelBenchmark] = None

    def __post_init__(self):
        self.specs = {}
        self.machine = self.machine.upper()

    def _log(self, *args):
        print(*args)
        # if self.debug:
        #     print(*args)

    def reduce_configs(self, max_kernels: int = None, seed: int = 42):
        self.configs = reduce_configs(self.configs, max_kernels, seed)

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
        print(
            f"Loading tuned config with average speedup of +{avg_speedup_percent:.2f}%"
        )

        self.configs = [
            (tag, config)
            for tag, config in self.configs
            if config.get_name() in tuned_data.keys()
        ]

    def save_results(self, results: List[BenchmarkResult]):
        if len(results) == 0:
            return None

        output_base = self.title or f"{self.kernel_type}_{self.backend}"

        output_csv_dir = Path(f"results/csv/{self.kernel_type}")
        output_json_dir = Path(f"results/json/{self.kernel_type}")

        output_csv_path = output_csv_dir / f"{output_base}.csv"
        output_json_path = output_json_dir / f"{output_base}.json"

        write_results_to_csv(results, output_csv_path)
        self._log(f"Results written to {output_csv_path}")

        write_results_to_json(results, output_json_path)
        self._log(f"Results written to {output_json_path}")

        return pd.read_csv(output_csv_path)

    def _create_benchmark(self, tag: str, config: OpConfig) -> KernelBenchmark:
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
        bench: KernelBenchmark = BENCHMARKS[self.kernel_type][self.backend](**kwargs)

        tuned_config = self.specs.get(config.get_name())
        if tuned_config:
            bench.load_tuned_config(tuned_config)

        return bench

    def _load_benches(self):
        """Create benchmark instances for all configurations."""
        self._benches = [
            self._create_benchmark(tag, config) for tag, config in self.configs
        ]

    @staticmethod
    def _compile_single_kernel_static(args):
        """Static method for parallel compilation."""
        tag, bench, hyperparams_dict, config_name = args
        try:
            # Get paths for this specific benchmark
            local_kernel_dir = bench.kernel_dir / bench.kernel_type / bench.backend
            mlir_dir = local_kernel_dir / "mlir"
            vmfb_dir = local_kernel_dir / "vmfb"

            os.makedirs(mlir_dir, exist_ok=True)
            os.makedirs(vmfb_dir, exist_ok=True)

            mlir_path = mlir_dir / f"{bench.config.get_name()}.mlir"
            vmfb_path = vmfb_dir / f"{bench.config.get_name()}.vmfb"

            # Set hyperparams if available
            if hasattr(bench, "hyperparams") and hyperparams_dict:
                bench.hyperparams = hyperparams_dict.get(config_name)

            success = bench.compile_to_vmfb(mlir_path, vmfb_path)
            return bench.config, vmfb_path, success
        except Exception as e:
            print(f"Compilation failed for {config_name}: {e}")
            return bench.config, None, False

    def _compile_kernels_parallel(self) -> Dict[str, Path]:
        """Compile all kernels in parallel."""
        if not self._benches:
            self._load_benches()

        iree_benches = [
            bench for bench in self._benches if isinstance(bench, IREEKernelBenchmark)
        ]

        if not iree_benches:
            return

        compilation_results = []
        compile_args = [
            (bench.tag, bench, self.hyperparams, bench.config.get_name())
            for bench in iree_benches
        ]
        if len(iree_benches) < 5:
            compilation_results = compilation_results = [
                self._compile_single_kernel_static(args) for args in compile_args
            ]
        else:
            num_cpus = max(1, cpu_count() - 20)
            self._log(f"Using {num_cpus} CPUs for parallel compilation.")

            with Pool(num_cpus) as pool:
                compilation_results = list(
                    tqdm(
                        pool.imap(self._compile_single_kernel_static, compile_args),
                        total=len(compile_args),
                        desc=f"Compiling {self.kernel_type.capitalize()} Kernels",
                    )
                )

        vmfb_dict = {}
        success_count = 0
        error_count = 0

        for result in compilation_results:
            config, vmfb_file, success = result
            config_name = config.get_name()
            if success:
                vmfb_dict[config_name] = vmfb_file
                success_count += 1
            else:
                error_count += 1

        self._log(
            f"{success_count} Success, {error_count} Failed out of {len(compilation_results)} configs"
        )
        self._log("Compilation process completed.")

        return vmfb_dict

    def benchmark_kernels(self):
        """Run benchmarks sequentially after parallel compilation."""
        self._load_benches()

        if len(self._benches) == 0:
            return

        use_iree = isinstance(self._benches[0], IREEKernelBenchmark)
        if use_iree:
            vmfb_dict = self._compile_kernels_parallel()
            bench_items = [
                bench for bench in self._benches if bench.config.get_name() in vmfb_dict
            ]
        else:
            bench_items = self._benches

        results = []

        for bench in tqdm(bench_items, desc="Benchmarking Kernels"):
            try:
                if use_iree:
                    iree_bench: IREEKernelBenchmark = bench
                    bench_result = iree_bench.bench_vmfb(
                        vmfb_filename=vmfb_dict.get(bench.config.get_name()),
                        device=self.device,
                        num_iterations=self.num_iterations,
                    )
                else:
                    bench_result = bench.run_bench(
                        device=self.device, num_iterations=self.num_iterations
                    )

                if not bench_result.ok:
                    tqdm.write(f"Benchmark failed for {bench.config.get_name()}")

                results.append(bench_result)

            except Exception as e:
                tqdm.write(f"Error benchmarking {bench.config.get_name()}: {e}")
                results.append(bench.get_bench_result(0, False))

        return self.save_results(results)

    def tune_scheduling(self, max_iterations=100):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/tuning/{self.kernel_type}")
        output_path = (
            output_dir / f"{self.kernel_type}_{self.backend}_schedule_tuned.json"
        )

        results = {}

        for tag, config in tqdm(self.configs, desc="Tuning Scheduling"):
            print(f"Tuning scheduling for kernel {config.get_name()}")
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
        self._load_benches()

        tuning_dir = Path(f"results/tuning")
        tuning_result_basename = f"{self.kernel_type}_{self.backend}_tuned_results.json"
        tuning_result_path = tuning_dir / self.kernel_type / tuning_result_basename

        tuning_paradigm = BayesianTuningParadigm()
        tuner = ParallelTuner(tuning_paradigm)
        tuner.tune_kernels(
            benches=self._benches,
            tuning_result_path=tuning_result_path,
            num_iterations=self.num_iterations,
            num_trials=num_trials,
            debug=self.debug,
            save_results=True,
        )

        print(f"Saved results to {tuning_result_path}")
