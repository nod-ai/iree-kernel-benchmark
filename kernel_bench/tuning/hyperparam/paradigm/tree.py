import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
import itertools

from kernel_bench.core.utils import BenchmarkResult
from kernel_bench.tuning.hyperparam.paradigm.paradigm import (
    TuningContext,
    TuningParadigm,
)
from kernel_bench.tuning.hyperparam.parameters import CategoricalBounds
from kernel_bench.utils.parallel import ProgressUpdate


@dataclass
class TreeParameter:
    """Represents a tunable parameter with integer range"""

    name: str
    min_value: int
    max_value: int
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
            # Initial pass: evenly distribute across full range
            if num_candidates == 1:
                return [(self.min_value + self.max_value) // 2]
            elif num_candidates >= (self.max_value - self.min_value + 1):
                # If we want more candidates than the range, return all values
                return list(range(self.min_value, self.max_value + 1))
            else:
                # Evenly distribute candidates
                step = (self.max_value - self.min_value) / (num_candidates - 1)
                candidates = []
                for i in range(num_candidates):
                    value = int(round(self.min_value + i * step))
                    candidates.append(value)
                # Remove duplicates while preserving order
                seen = set()
                unique_candidates = []
                for val in candidates:
                    if val not in seen:
                        seen.add(val)
                        unique_candidates.append(val)
                return unique_candidates
        else:
            # Refinement pass: search around the center point
            range_width = int((self.max_value - self.min_value) * range_fraction)
            # Ensure we have at least a range of num_candidates
            range_width = max(range_width, num_candidates - 1)

            local_min = max(self.min_value, center - range_width // 2)
            local_max = min(self.max_value, center + range_width // 2)

            # Adjust to ensure we have enough unique values
            while local_max - local_min + 1 < num_candidates:
                if local_min > self.min_value:
                    local_min -= 1
                if (
                    local_max < self.max_value
                    and local_max - local_min + 1 < num_candidates
                ):
                    local_max += 1
                if local_min == self.min_value and local_max == self.max_value:
                    break

            available_range = local_max - local_min + 1

            if num_candidates >= available_range:
                # Return all values in the range
                return list(range(local_min, local_max + 1))
            else:
                # Evenly distribute within the local range
                step = (local_max - local_min) / (num_candidates - 1)
                candidates = []
                for i in range(num_candidates):
                    value = int(round(local_min + i * step))
                    candidates.append(value)
                # Remove duplicates
                return list(dict.fromkeys(candidates))


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

    def get_name(self):
        return "Tree-based Multi-pass Tuning"

    def initialize(self, context: TuningContext):
        self.context = context
        self.tuning_spec = context.bench.tuning_spec
        tuning_params = self.tuning_spec.params()
        self.parameters = [
            TreeParameter(
                name=param.name,
                min_value=param.bounds.get_range()[0],
                max_value=param.bounds.get_range()[-1],
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

        pruned_configurations = []
        print("--- Pruned Configs ---")
        for config in configurations:
            sat, violated = self.tuning_spec.validate_constraints(config)
            if sat:
                print("*", end="")
                pruned_configurations.append(config)
            else:
                print("-", end="")
        print("---------------------")

        num_trials = self.context.num_trials
        max_configs = num_trials // self.num_passes + num_trials // 10
        if len(pruned_configurations) <= max_configs:
            return pruned_configurations

        pruned_configurations = list(
            np.random.choice(pruned_configurations, size=max_configs)
        )
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

        for config in unique_configs:
            score = self._benchmark(self.context, config)

            if (self.minimize and score < best_score) or (
                not self.minimize and score > best_score
            ):
                best_score = score
                best_config = config.copy()

            self.trial_count += 1
            self._update_progress(completed=self.trial_count)

        return best_config, best_score

    def _tune(self, context, progress_callback) -> BenchmarkResult:
        """
        Run the multi-pass tree-based tuning algorithm.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (best_configuration, best_score)
        """
        self.initialize(context)
        bench = context.bench

        verbose = False
        best_config = None
        best_score = None

        self.trial_count = 0

        max_configs_per_pass = int(self.num_candidates ** len(self.parameters))
        configs_per_pass = [
            max_configs_per_pass * self.num_passes for _ in range(self.num_passes)
        ]

        self._update_progress(completed=0, total=sum(configs_per_pass), active=True)

        for pass_num in range(self.num_passes):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Pass {pass_num + 1}/{self.num_passes}")
                print(f"{'='*50}")

            # Generate configurations for this pass
            configurations = self._generate_configurations(pass_num, best_config)
            for i in range(pass_num, self.num_passes):
                configs_per_pass[i] = len(configurations)

            self._update_progress(total=sum(configs_per_pass))

            if verbose:
                print(f"Generated {len(configurations)} valid configurations")
                if best_config:
                    print(f"Refining around: {best_config}")

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
                print(f"Pass {pass_num + 1} best score: {best_score}")
                print(f"Pass {pass_num + 1} best config: {best_config}")

        if verbose:
            print(f"\n{'='*50}")
            print(f"Tuning Complete!")
            print(f"Final best score: {best_score}")
            print(f"Final best configuration: {best_config}")
            print(
                f"Total evaluations: {sum(h['num_evaluations'] for h in self.history)}"
            )

        return best_score


# Example usage
if __name__ == "__main__":
    # Define a simple objective function
    def objective_function(config: Dict[str, int]) -> float:
        """Example objective function to minimize"""
        a = config["A"]
        b = config["B"]
        c = config["C"]
        d = config["D"]

        # Target: A=1, B=8, C=15, D=8
        return (a - 1) ** 2 + (b - 6) ** 2 + (c - 4) ** 2 + (d - 14) ** 2

    # Define parameters with integer ranges
    parameters = [
        TreeParameter("A", min_value=1, max_value=15),
        TreeParameter("B", min_value=1, max_value=15),
        TreeParameter("C", min_value=1, max_value=15),
        TreeParameter("D", min_value=1, max_value=15),
    ]

    # Create and run tuner
    tuner = MultiPassTreeTuner(
        parameters=parameters,
        objective_function=objective_function,
        num_candidates=2,
        num_passes=4,
        minimize=True,
        range_reduction_factor=0.3,  # Reduce range to 30% each pass
    )

    best_config, best_score = tuner.tune(verbose=True)
