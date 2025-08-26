import math
import os
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
import queue

import optuna
from wave_lang.kernel.wave.constraints import MMAType

from ...parallel import ParallelProgressManager
from ...bench_utils import OpConfig, get_kernel_perf_stats, TuningSpec


@dataclass
class TuningConstraint:
    """Represents a tuning parameter constraint."""

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
class TuningContext:
    """Context object containing all necessary information for tuning."""

    kernel: OpConfig
    config_tag: str
    device_id: int
    mfma_configs: List[Tuple[MMAType]]
    tiling_constraints: List[TuningConstraint]
    tuning_class: Type[TuningSpec]
    kernel_dir: os.PathLike
    num_iterations: int
    num_trials: int
    compile_kernel_func: Callable
    bench_kernel_func: Callable
    debug: bool = False
    worker_id: int = 0


@dataclass
class TuningResult:
    """Result of a tuning run."""

    runtime: float
    spec: Optional[TuningSpec]
    mfma_config: Optional[Tuple[MMAType]]
    config_name: str
    kernel: Optional[OpConfig] = None
    worker_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        if self.spec and self.mfma_config and self.kernel:
            arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
                self.kernel, self.runtime
            )
            return {
                "block_sizes": asdict(self.spec),
                "mfma_variant": (
                    [mfma.name for mfma in self.mfma_config]
                    if isinstance(self.mfma_config, (list, tuple))
                    else [self.mfma_config.name]
                ),
                "mean_microseconds": self.runtime,
                "arithmetic_intensity": arithmetic_intensity,
                "tflops": tflops_per_second,
                "problem": asdict(self.kernel),
            }
        return None


@dataclass
class ProgressUpdate:
    """Progress update message."""

    device_id: int
    completed: int
    total: int
    current: str
    active: bool
    is_final: bool = False
    worker_id: int = 0


@dataclass
class WorkerMessage:
    """Message from worker process."""

    type: str  # 'progress' or 'result'
    data: Any


class TuningParadigm(ABC):
    """Abstract base class for different tuning paradigms."""

    @abstractmethod
    def tune(self, context: TuningContext, progress_callback: Callable) -> TuningResult:
        """Run the tuning process and return the best result."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this tuning paradigm."""
        pass

    def _compile_and_benchmark(
        self,
        context: TuningContext,
        tuning_spec: TuningSpec,
        mfma_config: Tuple[MMAType],
    ) -> float:
        """Compile and benchmark a kernel configuration."""
        config_name = context.kernel.get_name()

        # Setup paths
        mlir_dir = context.kernel_dir / "mlir"
        vmfb_dir = context.kernel_dir / "vmfb"
        os.makedirs(mlir_dir, exist_ok=True)
        os.makedirs(vmfb_dir, exist_ok=True)

        mlir_path = mlir_dir / f"{config_name}.mlir"
        vmfb_path = vmfb_dir / f"{config_name}.vmfb"

        try:
            # Compile kernel
            compile_success = context.compile_kernel_func(
                context.kernel,
                mlir_path,
                vmfb_path,
                mfma_variant=mfma_config,
                spec=tuning_spec,
            )
            if not compile_success:
                raise Exception(f"Compiling kernel {config_name} failed")

            # Benchmark kernel
            runtime, bench_success = context.bench_kernel_func(
                context.kernel,
                vmfb_path,
                num_iterations=context.num_iterations,
                device=f"hip://{context.device_id}",
                debug=context.debug,
            )
            if not bench_success:
                raise Exception(f"Benchmarking kernel {config_name} failed")

            return runtime

        except Exception:
            return math.inf


def worker_process(
    context: TuningContext, tuning_paradigm: TuningParadigm, message_queue: Queue
):
    """Worker process function."""

    def progress_callback(update: ProgressUpdate):
        message_queue.put(WorkerMessage(type="progress", data=update))

    try:
        result = tuning_paradigm.tune(context, progress_callback)
        if result.spec:
            message_queue.put(WorkerMessage(type="result", data=result))
    except Exception as e:
        # Send final progress on error
        message_queue.put(
            WorkerMessage(
                type="progress",
                data=ProgressUpdate(
                    device_id=context.device_id,
                    completed=context.num_trials,
                    total=context.num_trials,
                    current=f"Error: {str(e)}",
                    active=False,
                    is_final=True,
                    worker_id=context.worker_id,
                ),
            )
        )


class ParallelTuner:
    """Manages parallel execution of tuning tasks."""

    def __init__(self, tuning_paradigm: TuningParadigm, num_gpus: Optional[int] = None):
        self.tuning_paradigm = tuning_paradigm
        self.num_gpus = num_gpus or 8
        self.results = {}
        self.results_lock = threading.Lock()

    def tune_kernels(
        self,
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
    ) -> Dict[str, Any]:
        """Run parallel tuning for multiple kernel configurations."""
        os.makedirs(os.path.dirname(tuning_result_path), exist_ok=True)

        total_configs = len(configs)
        num_workers = min(self.num_gpus, total_configs)

        # Setup progress manager
        progress_manager = ParallelProgressManager(total_configs, num_workers)
        progress_manager.start_main_progress()

        # Create message queue for worker communication
        message_queue = Queue()

        # Create and start worker processes
        processes = []
        for i, (config_tag, kernel) in enumerate(configs):
            device_id = i % num_workers
            context = TuningContext(
                kernel=kernel,
                config_tag=config_tag,
                device_id=device_id,
                mfma_configs=mfma_configs,
                tiling_constraints=tiling_constraints,
                tuning_class=tuning_class,
                kernel_dir=kernel_dir,
                num_iterations=num_iterations,
                num_trials=num_trials,
                compile_kernel_func=compile_kernel_func,
                bench_kernel_func=bench_kernel_func,
                debug=debug,
                worker_id=i,
            )

            p = Process(
                target=worker_process,
                args=(context, self.tuning_paradigm, message_queue),
            )
            p.start()
            processes.append(p)

        # Start threads for progress monitoring and display refresh
        stop_event = threading.Event()
        completed_workers = 0

        def message_handler():
            nonlocal completed_workers
            while not stop_event.is_set() or not message_queue.empty():
                try:
                    msg = message_queue.get(timeout=0.1)

                    if msg.type == "progress":
                        update = msg.data
                        progress_state = progress_manager.get_shared_state()
                        progress_state.update(
                            {
                                f"worker_{update.device_id}_completed": update.completed,
                                f"worker_{update.device_id}_total": update.total,
                                f"worker_{update.device_id}_current": update.current,
                                f"worker_{update.device_id}_active": update.active,
                            }
                        )

                        if update.is_final:
                            completed_workers += 1
                            progress_state["total_completed"] = completed_workers

                    elif msg.type == "result":
                        result = msg.data
                        self._save_result(result, tuning_result_path)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error handling message: {e}")

        def refresh_display():
            while not stop_event.is_set():
                progress_manager.refresh_display()
                time.sleep(0.5)

        # Start handler threads
        message_thread = threading.Thread(target=message_handler)
        refresh_thread = threading.Thread(target=refresh_display)
        message_thread.start()
        refresh_thread.start()

        try:
            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Give message handler time to process remaining messages
            time.sleep(1)

        finally:
            # Cleanup
            stop_event.set()
            message_thread.join()
            refresh_thread.join()
            progress_manager.close()

        # Save final results
        if save_results:
            with open(tuning_result_path, "w") as file:
                json.dump(self.results, file, indent=4)

        return self.results

    def _save_result(self, result: TuningResult, tuning_result_path: os.PathLike):
        """Save tuning result to file."""
        result_dict = result.to_dict()
        if result_dict:
            with self.results_lock:
                self.results[result.config_name] = result_dict
                with open(tuning_result_path, "w") as file:
                    json.dump(self.results, file, indent=4)
