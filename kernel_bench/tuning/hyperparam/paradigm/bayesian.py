import math
from typing import Callable, override
import optuna
from tqdm import tqdm
import torch

from ..parameters import CategoricalBounds, IntegerBounds
from .paradigm import TuningParadigm, TuningContext
from kernel_bench.utils.bench_utils import BenchmarkResult
from kernel_bench.utils.parallel import ProgressEvent


class BayesianTuningParadigm(TuningParadigm):
    """Optuna-based hyperparameter tuning implementation."""

    def get_name(self) -> str:
        return "Optuna NSGA-II"

    @override
    def _tune(
        self, context: TuningContext, progress_callback: Callable
    ) -> BenchmarkResult:
        """Run Optuna-based tuning."""

        bench = context.bench
        best_result = bench.get_bench_result(math.inf, False)
        trial_count = 0

        def objective(trial: optuna.Trial):
            nonlocal best_result, trial_count

            progress_callback(
                ProgressUpdate(
                    device_id=context.device_id,
                    completed=trial_count,
                    total=context.num_trials,
                    current=bench.config.get_name(),
                    active=True,
                    worker_id=context.worker_id,
                )
            )
            trial_count += 1

            params = bench.tuning_spec.params()
            integer_constraints = [
                (p.name, p.bounds)
                for p in params
                if isinstance(p.bounds, IntegerBounds)
            ]
            categorical_constraints = [
                (p.name, p.bounds)
                for p in params
                if isinstance(p.bounds, CategoricalBounds)
            ]

            # Suggest block sizes
            int_constraint_values = {
                name: trial.suggest_int(
                    name,
                    bounds.min,
                    bounds.max,
                    step=bounds.step,
                )
                for name, bounds in integer_constraints
            }
            cat_constraint_values = {
                name: trial.suggest_categorical(name, bounds.get_range())
                for name, bounds in categorical_constraints
            }
            constraint_vals = int_constraint_values | cat_constraint_values

            curr_result = self._benchmark(context, constraint_vals)

            if not curr_result.ok:
                raise optuna.TrialPruned()

            # Update best result if improved
            if curr_result.mean_microseconds < best_result.mean_microseconds:
                best_result = curr_result

            return curr_result.mean_microseconds

        def constraints_func(trial: optuna.trial.FrozenTrial):
            all_satisfied, violations = context.bench.validate_constraints(
                param_values=trial.params
            )
            return list(violations.values())

        # Configure and run Optuna study
        # optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(
            constraints_func=constraints_func,
            n_startup_trials=max(context.num_trials // 5, 15),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=context.num_trials,
            show_progress_bar=False,
        )

        # Send final progress update
        progress_callback(
            ProgressUpdate(
                device_id=context.device_id,
                completed=context.num_trials,
                total=context.num_trials,
                current="Idle",
                active=False,
                is_final=True,
                worker_id=context.worker_id,
            )
        )

        return best_result
