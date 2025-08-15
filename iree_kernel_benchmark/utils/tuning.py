import math
import os
import optuna
from dataclasses import asdict, dataclass
from multiprocessing import Pool, cpu_count, Manager
import threading
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
from wave_lang.kernel.wave.constraints import MMAType

from iree_kernel_benchmark.utils.wave_utils import TuningSpec
from .parallel import ParallelProgressManager
from .bench_utils import OpConfig, get_kernel_perf_stats


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


@dataclass
class TuningWorkerConfig:
    """Encapsulates all configuration needed for a tuning worker."""

    config: Tuple[str, OpConfig]
    device_id: int
    worker_id: int | str
    total_configs: int
    kernel_dir: os.PathLike
    num_iterations: int
    mfma_configs: List[Tuple[MMAType]]
    tiling_constraints: List[TuningConstraint]
    tuning_class: Type[TuningSpec]
    num_trials: int
    tuning_result_path: os.PathLike
    results_lock: threading.Lock
    shared_results: Any
    compile_kernel_func: Callable
    bench_kernel_func: Callable
    progress_state: Dict
    progress_lock: threading.Lock
    debug: Optional[bool]


def tune_config_worker(wargs: TuningWorkerConfig):
    """Worker function for parallel kernel tuning."""

    mfma_configs = wargs.mfma_configs
    tuning_result = (math.inf, None, mfma_configs[0])
    device_id = wargs.device_id
    config_tag, kernel = wargs.config
    config_name = kernel.get_name()

    trial_count = 0

    def update_progress(update: Dict[str, Any]):
        with wargs.progress_lock:
            wargs.progress_state.update(update)

    def objective(trial: optuna.Trial):
        nonlocal tuning_result, trial_count

        update_progress(
            {
                f"worker_{device_id}_completed": trial_count,
                f"worker_{device_id}_total": wargs.num_trials,
                f"worker_{device_id}_current": config_name,
                f"worker_{device_id}_active": True,
            }
        )

        trial_count += 1

        block_sizes = [
            trial.suggest_int(
                constraint.name,
                constraint.min,
                constraint.max,
                step=constraint.step,
            )
            for constraint in wargs.tiling_constraints
        ]

        tuning_spec_params = {
            constraint.name: block_sizes[i]
            for i, constraint in enumerate(wargs.tiling_constraints)
        }

        tuning_spec = wargs.tuning_class(**tuning_spec_params)

        mfma_config = mfma_configs[
            trial.suggest_categorical("MFMA_INDEX", list(range(len(mfma_configs))))
        ]

        mlir_dir = wargs.kernel_dir / "mlir"
        vmfb_dir = wargs.kernel_dir / "vmfb"
        os.makedirs(mlir_dir, exist_ok=True)
        os.makedirs(vmfb_dir, exist_ok=True)

        mlir_path = mlir_dir / f"{config_name}.mlir"
        vmfb_path = vmfb_dir / f"{config_name}.vmfb"

        try:
            compile_success = wargs.compile_kernel_func(
                kernel,
                mlir_path,
                vmfb_path,
                mfma_variant=mfma_config,
                spec=tuning_spec if wargs.num_trials > 0 else None,
            )
            if not compile_success:
                raise Exception(f"Compiling kernel {config_name} failed")
        except:
            return math.inf

        try:
            runtime, bench_success = wargs.bench_kernel_func(
                kernel,
                vmfb_path,
                num_iterations=wargs.num_iterations,
                device=f"hip://{device_id}",
                debug=wargs.debug,
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
        n_trials=wargs.num_trials,
        show_progress_bar=False,
    )

    best_runtime, best_spec, best_mfma = tuning_result

    update_progress(
        {
            f"worker_{device_id}_completed": wargs.num_trials,
            f"worker_{device_id}_active": False,
            f"worker_{device_id}_current": "Idle",
        }
    )
    with wargs.progress_lock:
        wargs.progress_state["total_completed"] += 1

    if best_spec and best_mfma:
        arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
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

        with wargs.results_lock:
            wargs.shared_results[config_name] = result
            with open(wargs.tuning_result_path, "w") as file:
                json.dump(dict(wargs.shared_results), file, indent=4)

        return config_name, result

    return config_name, None


def tune_kernels_parallel(
    configs: List[Tuple[str, OpConfig]],
    mfma_configs: List[Tuple[MMAType]],
    tiling_constraints: List[TuningConstraint],
    tuning_class: Type[TuningSpec],
    compile_kernel_func: Callable,
    bench_kernel_func: Callable,
    kernel_dir: os.PathLike,
    tuning_result_path: os.PathLike,
    num_iterations: int = 1,
    num_trials: int = 100,
    debug: bool = False,
    save_results: bool = True,
) -> dict[str, Any]:
    os.makedirs(os.path.dirname(tuning_result_path), exist_ok=True)

    manager = Manager()
    results_lock = manager.Lock()
    shared_results = manager.dict()

    worker_args = []
    total_configs = len(configs)
    num_gpus = min(8, total_configs)

    progress_manager = ParallelProgressManager(total_configs, num_gpus)
    progress_manager.start_main_progress()

    for i, config in enumerate(configs):
        device_id = i % num_gpus
        worker_args.append(
            TuningWorkerConfig(
                config=config,
                device_id=device_id,
                worker_id=i + 1,
                total_configs=total_configs,
                kernel_dir=kernel_dir,
                num_iterations=num_iterations,
                mfma_configs=mfma_configs,
                tiling_constraints=tiling_constraints,
                tuning_class=tuning_class,
                num_trials=num_trials,
                tuning_result_path=tuning_result_path,
                results_lock=results_lock,
                shared_results=shared_results,
                compile_kernel_func=compile_kernel_func,
                bench_kernel_func=bench_kernel_func,
                progress_state=progress_manager.get_shared_state(),
                progress_lock=progress_manager.get_lock(),
                debug=debug,
            )
        )

    stop_refresh = threading.Event()

    def refresh_thread():
        while not stop_refresh.is_set():
            progress_manager.refresh_display()
            time.sleep(20)

    refresh_thread_obj = threading.Thread(target=refresh_thread)
    refresh_thread_obj.start()

    try:
        with Pool(processes=num_gpus) as pool:
            for result in pool.imap_unordered(tune_config_worker, worker_args):
                if result[1] is not None:
                    pass
    finally:
        stop_refresh.set()
        refresh_thread_obj.join()
        progress_manager.close()

    result_dict = dict(shared_results)
    if save_results:
        with open(tuning_result_path, "w") as file:
            json.dump(result_dict, file, indent=4)
    return result_dict
