import os
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from multiprocessing import Process, Queue
import traceback
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
)
import queue
import torch
from tqdm import tqdm

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.print_utils import (
    get_logger,
    set_global_handler,
    set_tqdm_handler,
)
from .paradigm import TuningContext, TuningParadigm, TuningResult
from kernel_bench.utils.parallel_utils.progress_visualizer import (
    ParallelProgressManager,
)
from kernel_bench.utils.parallel_utils.progress_visualizer import WorkerMessage
from kernel_bench.utils.parallel_utils.progress_context import ProgressEvent


def worker_process(
    context: TuningContext, tuning_paradigm: TuningParadigm, message_queue: Queue
):
    """Worker process function."""

    def progress_callback(event: ProgressEvent):
        message_queue.put(WorkerMessage(type="progress", data=event))

    set_tqdm_handler()

    try:
        result = tuning_paradigm.tune(context, progress_callback)
        if result.hyperparams:
            message_queue.put(WorkerMessage(type="result", data=result))
    except Exception as e:
        get_logger().error("".join(traceback.format_exception(e)))
        # Send a final error event
        error_event = ProgressEvent(
            event_type="main_update",
            worker_id=context.worker_id,
            device_id=context.device_id,
            data={
                "completed": context.num_trials,
                "total": context.num_trials,
                "current": f"Error: {str(e)}",
                "active": False,
                "color": "red",
            },
        )
        message_queue.put(WorkerMessage(type="progress", data=error_event))


class ParallelTuner:
    """Manages parallel execution of tuning tasks."""

    def __init__(self, tuning_paradigm: TuningParadigm, num_gpus: Optional[int] = None):
        self.tuning_paradigm = tuning_paradigm
        self.num_gpus = num_gpus or 8
        self.results = {}
        self.results_lock = threading.Lock()
        self.logger = get_logger()

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
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        os.makedirs(os.path.dirname(tuning_result_path), exist_ok=True)

        total_configs = len(benches)
        num_workers = min(self.num_gpus, total_configs)

        # Setup progress manager
        progress_manager = ParallelProgressManager(total_configs, num_workers)
        progress_manager.start_main_progress()

        # Create message queue for worker communication
        message_queue = Queue()

        # Create work queue with all benchmarks
        work_queue = Queue()
        for i, bench in enumerate(benches):
            work_queue.put((i, bench))

        # Available GPU queue
        available_gpus = Queue()
        for gpu_id in range(num_workers):
            available_gpus.put(gpu_id)

        # Track active processes
        active_processes = {}
        processes_lock = threading.Lock()

        # Start threads for progress monitoring and display refresh
        stop_event = threading.Event()
        completed_workers = 0

        def worker_launcher():
            """Launch workers as GPUs become available."""
            while not stop_event.is_set():
                try:
                    # Check if there's work to do and a GPU available
                    if work_queue.empty():
                        break

                    # Wait for an available GPU (with timeout to check stop_event)
                    try:
                        gpu_id = available_gpus.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Get next work item
                    try:
                        worker_id, bench = work_queue.get_nowait()
                    except queue.Empty:
                        # No more work, return GPU to pool
                        available_gpus.put(gpu_id)
                        break

                    # Create context with assigned GPU
                    context = TuningContext(
                        bench=bench,
                        device_id=gpu_id,
                        num_iterations=num_iterations,
                        num_trials=num_trials,
                        debug=debug,
                        worker_id=worker_id,
                    )

                    # Start worker process
                    p = Process(
                        target=worker_process,
                        args=(context, self.tuning_paradigm, message_queue),
                    )
                    p.start()

                    with processes_lock:
                        active_processes[(worker_id, gpu_id)] = p

                except Exception as e:
                    self.logger.error(f"Error launching worker: {e}")

        def process_monitor():
            """Monitor processes and free GPUs when they complete."""
            nonlocal completed_workers
            while not stop_event.is_set() or active_processes:
                completed = []
                with processes_lock:
                    for (worker_id, gpu_id), process in active_processes.items():
                        if not process.is_alive():
                            process.join()
                            completed.append((worker_id, gpu_id))

                # Remove completed processes and free their GPUs
                for worker_id, gpu_id in completed:
                    with processes_lock:
                        del active_processes[(worker_id, gpu_id)]
                    available_gpus.put(gpu_id)
                    completed_workers += 1

                time.sleep(0.1)

        def message_handler():
            """Handle messages from worker processes."""
            worker_completed_count = 0
            while not stop_event.is_set() or not message_queue.empty():
                try:
                    msg: WorkerMessage = message_queue.get(timeout=0.1)

                    if msg.type == "progress":
                        event: ProgressEvent = msg.data

                        # Handle the progress event through the progress manager
                        progress_manager.handle_progress_event(event)

                        # Check if this is a final update (worker completed)
                        if (
                            event.event_type == "main_update"
                            and not event.data.get("active", True)
                            and event.data.get("completed", 0)
                            >= event.data.get("total", 1)
                        ):
                            worker_completed_count += 1

                    elif msg.type == "result":
                        result = msg.data
                        self._save_result(result, tuning_result_path)

                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")

        def refresh_display():
            """Refresh the progress display."""
            while not stop_event.is_set():
                progress_manager.refresh_display()
                time.sleep(0.5)

        # Start all handler threads
        launcher_thread = threading.Thread(target=worker_launcher)
        monitor_thread = threading.Thread(target=process_monitor)
        message_thread = threading.Thread(target=message_handler)
        refresh_thread = threading.Thread(target=refresh_display)

        launcher_thread.start()
        monitor_thread.start()
        message_thread.start()
        refresh_thread.start()

        try:
            # Wait for all work to be assigned
            launcher_thread.join()

            # Wait for all processes to complete
            while active_processes:
                time.sleep(0.1)

            # Give message handler time to process remaining messages
            time.sleep(1)

        finally:
            # Cleanup
            stop_event.set()
            monitor_thread.join()
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
