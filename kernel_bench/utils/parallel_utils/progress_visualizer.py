from dataclasses import dataclass
from multiprocessing import Manager
from typing import Any
from tqdm import tqdm

from kernel_bench.utils.print_utils import set_tqdm_handler
from .progress_context import ProgressEvent


@dataclass
class WorkerMessage:
    """Message from worker process."""

    type: str  # 'progress' or 'result'
    data: Any


class ParallelProgressManager:
    def __init__(self, total_configs: int, num_workers: int, pbar_offset: int = 0):
        """
        Initialize the parallel progress manager.

        Args:
            total_configs: Total number of configurations to process
            num_workers: Number of parallel workers
        """
        # Create manager for shared state
        self.manager = Manager()
        self.shared_state = self.manager.dict()
        self.lock = self.manager.Lock()

        # Initialize shared state
        self.shared_state["total_completed"] = 0
        self.shared_state["total_configs"] = total_configs

        # Initialize worker states
        for i in range(num_workers):
            self.shared_state[f"worker_{i}_completed"] = 0
            self.shared_state[f"worker_{i}_total"] = 0
            self.shared_state[f"worker_{i}_current"] = ""
            self.shared_state[f"worker_{i}_active"] = False
            self.shared_state[f"worker_{i}_color"] = "blue"
            self.shared_state[f"worker_{i}_sub_progress"] = self.manager.dict()

        self.num_workers = num_workers
        self.pbar_offset = pbar_offset
        self.pbar_gap = 1
        self.main_process_pbar = None
        self.worker_pbars: dict[int, tqdm] = {}
        self.sub_progress_pbars: dict[str, tqdm] = {}  # {worker_id}_{sub_id}: pbar

    def start_main_progress(self):
        """Start the main progress bar in the main process."""
        # Change logging
        set_tqdm_handler()

        # Create main progress bar
        self.main_process_pbar = tqdm(
            total=self.shared_state["total_configs"],
            desc="Overall Progress",
            position=0,
            colour="green",
        )

        # Create worker progress bars
        for i in range(self.num_workers):
            self.worker_pbars[i] = tqdm(
                total=100,  # Will be updated dynamically
                desc=f"\t- GPU {i}",
                position=self.pbar_gap * i + 1,
                colour=self.shared_state[f"worker_{i}_color"],
                ncols=200,
                leave=True,
            )

    def update_worker_assignment(
        self, worker_id: int, config_name: str, trials_total: int
    ):
        """Update when a worker is assigned a new configuration."""
        with self.lock:
            self.shared_state[f"worker_{worker_id}_completed"] = 0
            self.shared_state[f"worker_{worker_id}_total"] = trials_total
            self.shared_state[f"worker_{worker_id}_current"] = config_name
            self.shared_state[f"worker_{worker_id}_active"] = True

    def update_worker_progress(self, worker_id: int, trials_completed: int):
        """Update the progress of a specific worker."""
        with self.lock:
            self.shared_state[f"worker_{worker_id}_completed"] = trials_completed

    def complete_worker_task(self, worker_id: int):
        """Mark a worker's current task as complete."""
        with self.lock:
            self.shared_state["total_completed"] += 1
            self.shared_state[f"worker_{worker_id}_active"] = False
            self.shared_state[f"worker_{worker_id}_current"] = "Idle"

    def handle_progress_event(self, event: ProgressEvent):
        """Handle progress events from workers."""
        worker_id = event.device_id

        with self.lock:
            if event.event_type == "main_update":
                # Update main worker progress
                data = event.data
                self.shared_state[f"worker_{worker_id}_completed"] = data["completed"]
                self.shared_state[f"worker_{worker_id}_total"] = data["total"]
                self.shared_state[f"worker_{worker_id}_current"] = data["current"]
                self.shared_state[f"worker_{worker_id}_active"] = data["active"]
                self.shared_state[f"worker_{worker_id}_color"] = data["color"]

                # Check if this is a final update
                if not data["active"] and data["completed"] >= data["total"]:
                    self.shared_state["total_completed"] = (
                        self.shared_state.get("total_completed", 0) + 1
                    )

            elif event.event_type == "sub_create":
                # Create new sub-progress bar
                data = event.data
                sub_id = data["sub_id"]
                sub_key = f"{worker_id}_{sub_id}"

                worker_sub_progress = self.shared_state[
                    f"worker_{worker_id}_sub_progress"
                ]
                worker_sub_progress[sub_id] = {
                    "name": data["name"],
                    "total": data["total"],
                    "completed": data["completed"],
                    "color": data["color"],
                    "active": True,
                }
                self.shared_state[f"worker_{worker_id}_sub_progress"] = (
                    worker_sub_progress
                )

                num_worker_subs = len(
                    [
                        sub_key
                        for sub_key, bar in self.sub_progress_pbars.items()
                        if f"{worker_id}_" in sub_key
                    ]
                )

                # Create the actual progress bar
                if sub_key not in self.sub_progress_pbars:
                    self.sub_progress_pbars[sub_key] = tqdm(
                        total=data["total"],
                        desc=f"└─ {data['name']}",
                        position=worker_id * self.pbar_gap + num_worker_subs + 2,
                        colour=data["color"],
                        leave=True,
                        ncols=120,
                    )

            elif event.event_type == "sub_update":
                # Update existing sub-progress bar
                data = event.data
                sub_id = data["sub_id"]

                worker_sub_progress = self.shared_state[
                    f"worker_{worker_id}_sub_progress"
                ]
                if sub_id in worker_sub_progress:
                    sub_progress = dict(worker_sub_progress[sub_id])
                    sub_progress["completed"] = data["completed"]
                    if "total" in data:
                        sub_progress["total"] = data["total"]
                    worker_sub_progress[sub_id] = sub_progress
                    self.shared_state[f"worker_{worker_id}_sub_progress"] = (
                        worker_sub_progress
                    )

            elif event.event_type == "sub_complete":
                # Mark sub-progress as complete
                data = event.data
                sub_id = data["sub_id"]

                worker_sub_progress = self.shared_state[
                    f"worker_{worker_id}_sub_progress"
                ]
                if sub_id in worker_sub_progress:
                    sub_progress = dict(worker_sub_progress[sub_id])
                    sub_progress["completed"] = sub_progress["total"]
                    sub_progress["active"] = False
                    worker_sub_progress[sub_id] = sub_progress
                    self.shared_state[f"worker_{worker_id}_sub_progress"] = (
                        worker_sub_progress
                    )

            elif event.event_type == "sub_remove":
                # Remove sub-progress bar
                data = event.data
                sub_id = data["sub_id"]
                sub_key = f"{worker_id}_{sub_id}"

                worker_sub_progress = self.shared_state[
                    f"worker_{worker_id}_sub_progress"
                ]
                if sub_id in worker_sub_progress:
                    del worker_sub_progress[sub_id]
                    self.shared_state[f"worker_{worker_id}_sub_progress"] = (
                        worker_sub_progress
                    )

                if sub_key in self.sub_progress_pbars:
                    self.sub_progress_pbars[sub_key].close()
                    del self.sub_progress_pbars[sub_key]

    def refresh_display(self):
        """Refresh all progress bars based on shared state."""
        if self.main_process_pbar is None:
            return

        with self.lock:
            # Update main progress bar
            completed = self.shared_state.get("total_completed", 0)
            self.main_process_pbar.n = completed
            self.main_process_pbar.refresh()

            # Update worker progress bars
            for i in range(self.num_workers):
                if i in self.worker_pbars:
                    worker_pbar = self.worker_pbars[i]

                    completed = self.shared_state.get(f"worker_{i}_completed", 0)
                    total = self.shared_state.get(f"worker_{i}_total", 0)
                    color = self.shared_state.get(f"worker_{i}_color", "blue")

                    worker_pbar.total = total if total > 0 else 100
                    worker_pbar.n = completed
                    worker_pbar.colour = color
                    worker_pbar

                    if self.shared_state.get(f"worker_{i}_active", False):
                        current = self.shared_state.get(f"worker_{i}_current", "")
                        worker_pbar.set_description(f"GPU {i}: {current[:30]}")
                    else:
                        worker_pbar.set_description(f"GPU {i}: Idle")

                    worker_pbar.refresh()

                    # Update sub-progress bars for this worker
                    worker_sub_progress = self.shared_state.get(
                        f"worker_{i}_sub_progress", {}
                    )
                    for sub_id, sub_data in worker_sub_progress.items():
                        sub_key = f"{i}_{sub_id}"
                        if sub_key in self.sub_progress_pbars:
                            sub_pbar = self.sub_progress_pbars[sub_key]
                            sub_pbar.total = sub_data.get("total", 100)
                            sub_pbar.n = sub_data.get("completed", 0)
                            sub_pbar.colour = sub_data.get("color", "cyan")

                            status = (
                                "Active" if sub_data.get("active", True) else "Complete"
                            )
                            sub_pbar.set_description(
                                f"\t\t└─ {sub_data.get('name', 'Unknown')} ({status})"
                            )
                            sub_pbar.refresh()

    def close(self):
        """Close all progress bars."""
        if self.main_process_pbar:
            self.main_process_pbar.close()
        for pbar in self.worker_pbars.values():
            pbar.close()
        for pbar in self.sub_progress_pbars.values():
            pbar.close()

    def get_shared_state(self):
        """Get the shared state dictionary for passing to workers."""
        return self.shared_state

    def get_lock(self):
        """Get the lock for synchronization."""
        return self.lock
