import math
import os
import optuna
from dataclasses import asdict, dataclass
import multiprocessing as mp
import json
from .parallel import ParallelProgressManager


@dataclass
class TuningConstraint:
    name: str
    min: int
    max: int
    step: int
    exponential: bool = False

    def get_range(self) -> list[int]:
        range = []

        curr = self.min
        while curr <= self.max:
            range.append(curr)
            if self.exponential:
                curr *= self.step
            else:
                if curr == 1 and self.step > 1:
                    curr += self.step - 1
                else:
                    curr += self.step

        return range


def tune_config_worker(args):
    """Worker function for parallel kernel tuning."""
    (
        config,
        device_id,
        worker_id,
        total,
        kernel_type,
        backend,
        kernel_dir,
        num_iterations,
        debug,
        mfma_configs,
        tiling_constraints,
        tuning_class,
        num_trials,
        tuning_result_path,
        results_lock,
        shared_results,
        compile_kernel_func,
        bench_kernel_func,
        get_kernel_perf_stats_func,
        progress_state,  # Add this
        progress_lock,  # Add this
    ) = args

    tuning_result = (math.inf, None, mfma_configs[0])
    config_tag, kernel = config
    config_name = kernel.get_name()

    trial_count = 0

    def objective(trial: optuna.Trial):
        nonlocal tuning_result, trial_count

        with progress_lock:
            progress_state[f"worker_{device_id}_completed"] = trial_count
            progress_state[f"worker_{device_id}_total"] = num_trials
            progress_state[f"worker_{device_id}_current"] = config_name
            progress_state[f"worker_{device_id}_active"] = True

        trial_count += 1

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

        tuning_spec = tuning_class(**tuning_spec_params)

        mfma_config = mfma_configs[
            trial.suggest_categorical("MFMA_INDEX", list(range(len(mfma_configs))))
        ]

        local_kernel_dir = kernel_dir / kernel_type / backend
        mlir_dir = local_kernel_dir / "mlir"
        vmfb_dir = local_kernel_dir / "vmfb"
        os.makedirs(mlir_dir, exist_ok=True)
        os.makedirs(vmfb_dir, exist_ok=True)

        mlir_path = mlir_dir / f"{config_name}.mlir"
        vmfb_path = vmfb_dir / f"{config_name}.vmfb"

        try:
            compile_success = compile_kernel_func(
                kernel,
                mlir_path,
                vmfb_path,
                mfma_variant=mfma_config,
                spec=tuning_spec if num_trials > 0 else None,
            )
            if not compile_success:
                raise Exception(f"Compiling kernel {config_name} failed")
        except:
            return math.inf

        try:
            runtime, bench_success = bench_kernel_func(
                kernel,
                vmfb_path,
                num_iterations=num_iterations,
                device=f"hip://{device_id}",
                debug=debug,
            )
            if not bench_success:
                raise Exception(f"Benchmarking kernel {config_name} failed")
        except:
            return math.inf

        if runtime < tuning_result[0]:
            tuning_result = (runtime, tuning_spec, mfma_config)

        return runtime

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=num_trials,
        show_progress_bar=False,
    )

    best_runtime, best_spec, best_mfma = tuning_result

    # Mark task as complete
    with progress_lock:
        progress_state["total_completed"] += 1
        progress_state[f"worker_{device_id}_completed"] = num_trials
        progress_state[f"worker_{device_id}_active"] = False
        progress_state[f"worker_{device_id}_current"] = "Idle"

    if best_spec and best_mfma:
        arithmetic_intensity, tflops_per_second = get_kernel_perf_stats_func(
            kernel, best_runtime
        )

        result = {
            "block_sizes": asdict(best_spec),
            "mfma_variant": (
                [mfma.name for mfma in best_mfma]
                if isinstance(best_mfma, (list, tuple))
                else [best_mfma.name]
            ),
            "mean_microseconds": best_runtime,
            "arithmetic_intensity": arithmetic_intensity,
            "tflops": tflops_per_second,
            "problem": asdict(kernel),
        }

        with results_lock:
            shared_results[config_name] = result
            with open(tuning_result_path, "w") as file:
                json.dump(dict(shared_results), file, indent=4)

        return config_name, result

    return config_name, None
