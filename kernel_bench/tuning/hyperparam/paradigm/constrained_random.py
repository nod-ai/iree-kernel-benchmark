import copy
import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass, replace
from dataclass_wizard import asdict
import itertools

from kernel_bench.core.base import create_benchmark
from kernel_bench.core.template import (
    IREEKernelBenchmark,
    KernelBenchmark,
    batch_benchmark,
    batch_compile_iree_benches,
)
from kernel_bench.utils.bench_utils import BenchmarkResult
from kernel_bench.tuning.hyperparam.paradigm.paradigm import (
    TuningContext,
    TuningParadigm,
)
from kernel_bench.tuning.hyperparam.parameters import CategoricalBounds
from kernel_bench.utils.print_utils import get_logger
from kernel_bench.utils.parallel_utils.progress_visualizer import ProgressEvent
from kernel_bench.utils.parallel_utils.progress_context import MainProgress


class ConstrainedRandomTuner(TuningParadigm):

    def __init__(self):
        super().__init__()
        self.logger = get_logger()

    def get_name(self):
        return "Constrained Random Tuning Paradigm"

    def initialize(self, context: TuningContext, progress: MainProgress):
        self.context = context
        self.progress = progress
        self.tuning_spec = context.bench.tuning_spec
        self.params = self.tuning_spec.params()
        self.max_trials = context.num_trials

    def _generate_candidates(self, num_candidates: int) -> List[KernelBenchmark]:
        candidates = []

        while len(candidates) < num_candidates:
            config = {p.name: random.choice(p.bounds.get_range()) for p in self.params}
            new_bench = copy.deepcopy(self.context.bench)
            new_bench.update_parameter_values(config)

            is_valid, _ = new_bench.tuning_spec.validate_constraints()
            if is_valid:
                candidates.append(new_bench)

        return candidates

    def _tune(self, context, progress) -> BenchmarkResult:
        self.bench_results = []

        self.initialize(context, progress)

        best_config = None
        best_score = float("inf")

        num_attempts = 0
        batch_size = max(20, self.max_trials // 4)

        self.progress.update(
            completed=0,
            total=self.max_trials,
            current="Benchmarking",
        )

        def compile_callback(compile_res):
            if self.progress.current != "Compiling":
                self.progress.configure(
                    total=batch_size, description="Compiling", color="yellow"
                )
                self.progress.update(num_attempts)
            self.progress.step()

        def bench_callback(bench_res):
            if self.progress.current != "Benchmarking":
                self.progress.configure(
                    total=self.max_trials, description="Benchmarking", color="blue"
                )
                self.progress.update(num_attempts)
            if bench_res.ok:
                self.progress.step()

        while num_attempts < self.max_trials:
            # Generate & benchmark candidates
            candidates = self._generate_candidates(batch_size)
            bench_results = batch_benchmark(
                candidates,
                device=f"hip://{self.context.device_id}",
                num_iterations=context.num_iterations,
                timeout=self.base_exec_time * 3 if self.base_exec_time else None,
                compile_callback=compile_callback,
                bench_callback=bench_callback,
                validate_numerics=False,
                verbose=True,
                unique_ids=True,
            )

            # Update best score
            for candidate, bench_res in zip(candidates, bench_results):
                if not bench_res.ok:
                    continue
                self.bench_results.append(
                    {
                        "improvement": bench_res < self.base_result,
                        "runtime": bench_res.mean_microseconds,
                        "tflops": bench_res.tflops,
                        "config": bench_res.tuning_config,
                        "success": bench_res.ok,
                    }
                )
                if bench_res < best_score:
                    best_score = bench_res
                    best_config = candidate.tuning_spec.to_dict()

                    if bench_res < self.base_result:
                        config_name = context.bench.config.get_name()
                        percent_improvement = (
                            bench_res.tflops / best_score.tflops - 1
                        ) * 100
                        self.logger.info(
                            f"{config_name}: Found improvement of {percent_improvement:.2f}% \n"
                            f"{json.dumps(best_config, indent=4)}"
                        )

                num_attempts += 1

        os.makedirs("results/tuning/history", exist_ok=True)
        with open(
            f"results/tuning/history/{context.bench.backend}_{context.bench.config.get_name()}.json",
            "w",
        ) as file:
            json.dump(
                {
                    "config": context.bench.config.to_dict(),
                    "history": self.bench_results,
                },
                file,
                indent=4,
            )

        return best_score
