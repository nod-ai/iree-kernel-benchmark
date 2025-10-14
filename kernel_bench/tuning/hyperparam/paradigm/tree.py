import copy
import json
import os
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


@dataclass
class TreeParameter:
    """Represents a tunable parameter with a list of acceptable values"""

    name: str
    values: List[int]
    is_categorical: bool = False

    def get_candidates(
        self,
        num_candidates: int,
        center: Optional[int] = None,
        range_fraction: float = 1.0,
    ) -> List[int]:
        """
        Generate candidate integer values for this parameter.

        Args:
            num_candidates: Number of candidates to generate
            center: Center point for refined search (None for initial pass)
            range_fraction: Fraction of the original range to use (for refinement)
        """

        if center is None:
            # Initial pass: evenly distribute across all values
            if num_candidates == 1:
                # Return the middle value
                return [self.values[len(self.values) // 2]]
            elif num_candidates >= len(self.values):
                # If we want more candidates than available values, return all values
                return self.values.copy()
            else:
                # Evenly distribute candidates across the available values
                indices = []
                step = (len(self.values) - 1) / (num_candidates - 1)
                for i in range(num_candidates):
                    idx = int(round(i * step))
                    indices.append(idx)
                # Remove duplicate indices while preserving order
                unique_indices = list(dict.fromkeys(indices))
                return [self.values[i] for i in unique_indices]
        else:
            # Refinement pass: search around the center point
            if center not in self.values:
                # Find the closest value to center
                center_idx = min(
                    range(len(self.values)), key=lambda i: abs(self.values[i] - center)
                )
            else:
                center_idx = self.values.index(center)

            # Calculate the range of indices to consider
            total_range = len(self.values) - 1
            search_range = max(1, int(total_range * range_fraction))

            # Ensure we have at least num_candidates range if possible
            search_range = max(search_range, num_candidates - 1)

            # Calculate local bounds
            local_min_idx = max(0, center_idx - search_range // 2)
            local_max_idx = min(len(self.values) - 1, center_idx + search_range // 2)

            # Adjust to ensure we have enough unique values
            while local_max_idx - local_min_idx + 1 < num_candidates:
                if local_min_idx > 0:
                    local_min_idx -= 1
                if (
                    local_max_idx < len(self.values) - 1
                    and local_max_idx - local_min_idx + 1 < num_candidates
                ):
                    local_max_idx += 1
                if local_min_idx == 0 and local_max_idx == len(self.values) - 1:
                    break

            available_values = self.values[local_min_idx : local_max_idx + 1]

            if num_candidates >= len(available_values):
                # Return all values in the range
                return available_values
            else:
                # Evenly distribute within the local range
                indices = []
                step = (len(available_values) - 1) / (num_candidates - 1)
                for i in range(num_candidates):
                    idx = int(round(i * step))
                    indices.append(idx)
                # Remove duplicates
                unique_indices = list(dict.fromkeys(indices))
                return [available_values[i] for i in unique_indices]


class MultiPassTreeTuner(TuningParadigm):
    """Multi-pass tree-based hyperparameter tuner for integer parameters"""

    def __init__(
        self,
        num_candidates: int = 3,
        num_passes: int = 3,
        minimize: bool = True,
        range_reduction_factor: float = 0.5,
    ):
        """
        Initialize the tuner.

        Args:
            parameters: List of IntParameter objects defining the search space
            objective_function: Function to evaluate a configuration (returns score)
            num_candidates: Number of candidates to try for each parameter
            num_passes: Number of refinement passes
            minimize: Whether to minimize (True) or maximize (False) the objective
            range_reduction_factor: Factor by which to reduce search range each pass
        """
        super().__init__()
        self.num_candidates = num_candidates
        self.num_passes = num_passes
        self.minimize = minimize
        self.range_reduction_factor = range_reduction_factor
        self.history = []
        self.parameters = []
        self.logger = get_logger()

    def get_name(self):
        return "Tree-based Multi-pass Tuning"

    def initialize(self, context: TuningContext, progress: MainProgress):
        self.context = context
        self.progress = progress
        self.tuning_spec = context.bench.tuning_spec
        tuning_params = self.tuning_spec.params()
        self.parameters = [
            TreeParameter(
                name=param.name,
                values=param.bounds.get_range(),
                is_categorical=isinstance(param.bounds, CategoricalBounds),
            )
            for param in tuning_params
        ]

    def _generate_configurations(
        self, pass_num: int, best_config: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, int]]:
        """
        Generate all configurations for a given pass.

        Args:
            pass_num: Current pass number (0-indexed)
            best_config: Best configuration from previous pass
        """
        # Calculate the range fraction for this pass
        range_fraction = self.range_reduction_factor**pass_num if pass_num > 0 else 1.0

        # Generate candidates for each parameter
        all_candidates = []
        for param in self.parameters:
            if best_config is None or param.is_categorical:
                # First pass or categorical variable: use full range
                candidates = param.get_candidates(self.num_candidates)
            else:
                # Subsequent passes for integer variables: refine around best value
                center = best_config[param.name]
                candidates = param.get_candidates(
                    self.num_candidates, center, range_fraction
                )
            all_candidates.append([(param.name, val) for val in candidates])

        # Generate all combinations using tree structure
        configurations = []
        for combination in itertools.product(*all_candidates):
            config = dict(combination)
            configurations.append(config)

        pruned_configurations = [
            config
            for config in configurations
            if self.tuning_spec.validate_constraints(config)[0]
        ]

        # num_trials = self.context.num_trials
        # max_configs = num_trials // self.num_passes + num_trials // 10
        # if len(pruned_configurations) <= max_configs:
        #     return pruned_configurations

        # pruned_configurations = list(
        #     np.random.choice(pruned_configurations, size=max_configs)
        # )
        return pruned_configurations

    def _evaluate_configurations(
        self, configurations: List[Dict[str, int]]
    ) -> Tuple[Dict[str, int], float]:
        """
        Evaluate all configurations and return the best one.

        Args:
            configurations: List of configurations to evaluate

        Returns:
            Tuple of (best_configuration, best_score)
        """
        best_config = None
        best_score = float("inf") if self.minimize else float("-inf")

        # Remove duplicate configurations
        unique_configs = []
        seen = set()
        for config in configurations:
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)

        base_bench = self.context.bench

        batch_benches: List[KernelBenchmark] = []
        for config in unique_configs:
            candidate_bench = copy.deepcopy(base_bench)
            candidate_bench.update_parameter_values(config)
            batch_benches.append(candidate_bench)

        old_total = self.progress.total
        self.progress.configure(
            total=len(batch_benches), description=f"Compiling", color="yellow"
        )

        def compile_callback(compile_res):
            self.progress.update(increment=1)

        def bench_callback(bench_res: BenchmarkResult):
            nonlocal old_total
            if old_total:
                self.progress.configure(
                    total=old_total, description=f"Benchmarking", color="blue"
                )
                old_total = None

            self.bench_results.append(
                {
                    "config": bench_res.tuning_config,
                    "success": bench_res.ok,
                }
            )

            self.progress.update(increment=1)

        scores = batch_benchmark(
            batch_benches,
            device=f"hip://{self.context.device_id}",
            num_iterations=self.context.num_iterations,
            timeout=self.base_exec_time * 3 if self.base_exec_time else None,
            compile_callback=compile_callback,
            bench_callback=bench_callback,
            validate_numerics=False,
            verbose=True,
            unique_ids=True,
        )

        for i, score in enumerate(scores):
            if (self.minimize and score < best_score) or (
                not self.minimize and score > best_score
            ):
                best_score = score
                best_config = unique_configs[i].copy()

        return best_config, best_score

    def _tune(self, context, progress) -> BenchmarkResult:
        """
        Run the multi-pass tree-based tuning algorithm.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (best_configuration, best_score)
        """
        self.bench_results = []

        self.initialize(context, progress)
        bench = context.bench

        verbose = False
        best_config = None
        best_score = None

        self.trial_count = 0

        max_configs_per_pass = int(self.num_candidates ** len(self.parameters))
        configs_per_pass = [
            max_configs_per_pass * self.num_passes for _ in range(self.num_passes)
        ]

        self.progress.update(
            total=sum(configs_per_pass),
            current="Benchmarking",
        )

        for pass_num in range(self.num_passes):
            if verbose:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Pass {pass_num + 1}/{self.num_passes}")
                self.logger.info(f"{'='*50}")

            # Generate configurations for this pass
            configurations = self._generate_configurations(pass_num, best_config)
            for i in range(pass_num, self.num_passes):
                configs_per_pass[i] = len(configurations)

            self.progress.update(total=sum(configs_per_pass))

            if verbose:
                self.logger.info(
                    f"Generated {len(configurations)} valid configurations"
                )
                if best_config:
                    self.logger.info(f"Refining around: {best_config}")

            # Evaluate all configurations
            pass_best_config, pass_best_score = self._evaluate_configurations(
                configurations
            )

            # Update global best
            if (
                best_score is None
                or (self.minimize and pass_best_score < best_score)
                or (not self.minimize and pass_best_score > best_score)
            ):
                best_config = pass_best_config
                best_score = pass_best_score

            # Store history
            self.history.append(
                {
                    "pass": pass_num + 1,
                    "best_config": best_config.copy(),
                    "best_score": best_score,
                    "num_evaluations": len(configurations),
                }
            )

            if verbose:
                self.logger.info(f"Pass {pass_num + 1} best score: {best_score}")
                self.logger.info(f"Pass {pass_num + 1} best config: {best_config}")

        if verbose:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Tuning Complete!")
            self.logger.info(f"Final best score: {best_score}")
            self.logger.info(f"Final best configuration: {best_config}")
            self.logger.info(
                f"Total evaluations: {sum(h['num_evaluations'] for h in self.history)}"
            )

        os.makedirs("results/tuning/history", exist_ok=True)
        with open(
            f"results/tuning/history/{context.bench.kernel_type}_{context.bench.backend}_{context.device_id}.json",
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
