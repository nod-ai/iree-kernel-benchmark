from .parameters import CategoricalBounds, IntegerBounds
from .parallel_tuning import *
import math
from typing import Callable
import optuna


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

            curr_result = self._benchmark(
                context, int_constraint_values | cat_constraint_values
            )

            # Update best result if improved
            if curr_result.mean_microseconds < best_result.mean_microseconds:
                best_result = curr_result

            return curr_result.mean_microseconds

        # Configure and run Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")

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
