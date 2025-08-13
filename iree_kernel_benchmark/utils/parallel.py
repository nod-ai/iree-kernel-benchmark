from multiprocessing import Manager, Lock
from tqdm import tqdm
import sys
from typing import Dict, Optional
import time


class ParallelProgressManager:
    def __init__(self, total_configs: int, num_workers: int):
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

        self.num_workers = num_workers
        self.main_process_pbar = None
        self.worker_pbars: dict[int, tqdm] = {}

    def start_main_progress(self):
        """Start the main progress bar in the main process."""
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
                desc=f"GPU {i}",
                position=i + 1,
                colour="blue",
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

    def refresh_display(self):
        """Refresh all progress bars based on shared state."""
        if self.main_process_pbar is None:
            return

        with self.lock:
            # Update main progress bar
            completed = self.shared_state["total_completed"]
            self.main_process_pbar.n = completed
            self.main_process_pbar.refresh()

            # Update worker progress bars
            for i in range(self.num_workers):
                if i in self.worker_pbars:
                    worker_pbar = self.worker_pbars[i]

                    completed = self.shared_state[f"worker_{i}_completed"]
                    total = self.shared_state[f"worker_{i}_total"]

                    worker_pbar.total = total
                    worker_pbar.n = completed

                    if self.shared_state[f"worker_{i}_active"]:
                        current = self.shared_state[f"worker_{i}_current"]
                        worker_pbar.set_description(f"GPU {i}: {current[:30]}")
                    else:
                        worker_pbar.set_description(f"GPU {i}: Idle")

                    worker_pbar.refresh()

    def close(self):
        """Close all progress bars."""
        if self.main_process_pbar:
            self.main_process_pbar.close()
        for pbar in self.worker_pbars.values():
            pbar.close()

    def get_shared_state(self):
        """Get the shared state dictionary for passing to workers."""
        return self.shared_state

    def get_lock(self):
        """Get the lock for synchronization."""
        return self.lock
