import os
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Callable,
    Type,
    override,
)
import queue

from ...template import KernelBenchmark
from ...parallel import ParallelProgressManager
from ...bench_utils import BenchmarkResult, OpConfig


@dataclass
class TuningContext:
    """Context object containing all necessary information for tuning."""

    bench: KernelBenchmark
    device_id: int
    num_iterations: int
    num_trials: int
    debug: bool = False
    worker_id: int = 0


@dataclass
class TuningResult:
    """Result of a tuning run."""

    name: str
    benchmark: BenchmarkResult
    improvement: bool
    speedup: float
    hyperparams: Optional[Dict[str, Any]] = None


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

    def tune(self, context: TuningContext, progress_callback: Callable) -> TuningResult:
        """Run the tuning process and return the best result."""
        config = context.bench.config

        base_result = self._benchmark(context)
        tuned_result = self._tune(context, progress_callback)

        base_runtime = base_result.mean_microseconds
        tuned_runtime = tuned_result.mean_microseconds

        if not tuned_result.ok or base_runtime < tuned_runtime:
            best_result = base_result
            improvement = False
            speedup = 0
        else:
            best_result = tuned_result
            improvement = True
            speedup = base_runtime / tuned_runtime

        return TuningResult(
            name=config.get_name(),
            benchmark=best_result,
            improvement=improvement,
            speedup=speedup,
            hyperparams=best_result.tuning_config,
        )

    @abstractmethod
    def _tune(
        self, context: TuningContext, progress_callback: Callable
    ) -> BenchmarkResult:
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this tuning paradigm."""
        pass

    def _benchmark(
        self,
        context: TuningContext,
        param_values: Optional[Dict[str, int]] = None,
    ) -> BenchmarkResult:
        """Compile and benchmark a kernel configuration."""
        bench = context.bench

        bench.tuning_spec.clear()
        if param_values:
            for name, val in param_values.items():
                bench.tuning_spec.set_parameter(name, val)

        bench_result = bench.run_bench(
            f"hip://{context.device_id}", context.num_iterations
        )
        return bench_result


def worker_process(
    context: TuningContext, tuning_paradigm: TuningParadigm, message_queue: Queue
):
    """Worker process function."""

    def progress_callback(update: ProgressUpdate):
        message_queue.put(WorkerMessage(type="progress", data=update))

    try:
        result = tuning_paradigm.tune(context, progress_callback)
        if result.hyperparams:
            message_queue.put(WorkerMessage(type="result", data=result))
    except Exception as e:
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
        benches: List[KernelBenchmark],
        tuning_result_path: os.PathLike,
        num_iterations: int = 1,
        num_trials: int = 100,
        debug: bool = False,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Run parallel tuning for multiple kernel configurations."""
        os.makedirs(os.path.dirname(tuning_result_path), exist_ok=True)

        total_configs = len(benches)
        num_workers = min(self.num_gpus, total_configs)

        # Setup progress manager
        progress_manager = ParallelProgressManager(total_configs, num_workers)
        progress_manager.start_main_progress()

        # Create message queue for worker communication
        message_queue = Queue()

        # Create and start worker processes
        processes = []
        for i, bench in enumerate(benches):
            device_id = i % num_workers
            context = TuningContext(
                bench=bench,
                device_id=device_id,
                num_iterations=num_iterations,
                num_trials=num_trials,
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
                    msg: WorkerMessage = message_queue.get(timeout=0.1)

                    if msg.type == "progress":
                        update: ProgressUpdate = msg.data
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
        result_dict = asdict(result)
        if result_dict:
            with self.results_lock:
                self.results[result.name] = result_dict
                with open(tuning_result_path, "w") as file:
                    json.dump(self.results, file, indent=4)
