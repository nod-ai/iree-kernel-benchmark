from .utils import *
import math
from typing import Callable
import optuna


class BayesianTuningParadigm(TuningParadigm):
    """Optuna-based hyperparameter tuning implementation."""

    def get_name(self) -> str:
        return "Optuna NSGA-II"

    def tune(self, context: TuningContext, progress_callback: Callable) -> TuningResult:
        """Run Optuna-based tuning."""
        best_result = TuningResult(
            runtime=math.inf,
            spec=None,
            mfma_config=context.mfma_configs[0],
            config_name=context.kernel.get_name(),
            kernel=context.kernel,
            worker_id=context.worker_id,
        )

        trial_count = 0

        def objective(trial: optuna.Trial):
            nonlocal best_result, trial_count

            # Update progress
            progress_callback(
                ProgressUpdate(
                    device_id=context.device_id,
                    completed=trial_count,
                    total=context.num_trials,
                    current=context.kernel.get_name(),
                    active=True,
                    worker_id=context.worker_id,
                )
            )
            trial_count += 1

            # Suggest block sizes
            block_sizes = [
                trial.suggest_int(
                    constraint.name,
                    constraint.min,
                    constraint.max,
                    step=constraint.step,
                )
                for constraint in context.tiling_constraints
            ]

            # Create tuning spec
            tuning_spec_params = {
                constraint.name: block_sizes[i]
                for i, constraint in enumerate(context.tiling_constraints)
            }
            tuning_spec = context.tuning_class(**tuning_spec_params)

            # Check shared memory constraint
            shared_mem_constraint = (
                context.kernel.get_shared_mem_bytes(tuning_spec) - 65536
            )
            trial.set_user_attr("constraint", (shared_mem_constraint,))

            # Select MFMA config
            mfma_index = trial.suggest_categorical(
                "MFMA_INDEX", list(range(len(context.mfma_configs)))
            )
            mfma_config = context.mfma_configs[mfma_index]

            # Compile and benchmark
            runtime = self._compile_and_benchmark(context, tuning_spec, mfma_config)

            # Update best result if improved
            if runtime < best_result.runtime:
                best_result = TuningResult(
                    runtime=runtime,
                    spec=tuning_spec,
                    mfma_config=mfma_config,
                    config_name=context.kernel.get_name(),
                    kernel=context.kernel,
                    worker_id=context.worker_id,
                )

            return runtime

        def constraints(trial: optuna.Trial):
            return trial.user_attrs["constraint"]

        # Configure and run Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Add default tuning specs
        for i in range(len(context.mfma_configs)):
            study.enqueue_trial(
                {
                    **context.tuning_class().hyperparams(),
                    "MFMA_INDEX": i,
                }
            )

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
