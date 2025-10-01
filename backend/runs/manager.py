import logging
from typing import List, Optional

from backend.github_utils.auth import get_repo
from backend.runs import RunType
from backend.runs.run_utils import find_incomplete_runs
from backend.runs.tracker import get_run_tracker
from backend.storage.auth import get_blob_client
from backend.storage.types import WorkflowRunState

logger = logging.getLogger(__name__)


class RunManager:
    def __init__(self, runs: Optional[List[WorkflowRunState]] = None):
        if not runs:
            runs = []
        self._trackers = {run._id: get_run_tracker(run) for run in runs}
        self._dir_client = get_blob_client()
        self.load_incomplete_runs()

    def load_incomplete_runs(self):
        runs = find_incomplete_runs()
        for run in runs:
            self.track_run(run)

    def track_run(self, run: WorkflowRunState):
        if run._id not in self._trackers:
            self._trackers[run._id] = get_run_tracker(run)

    def remove_run(self, run_id: str):
        del self._trackers[run_id]

    def update_runs(self):
        self.load_incomplete_runs()
        completed_runs = []

        for run_id, tracker in self._trackers.items():
            if not tracker.is_complete():
                logger.debug(f"Updating ongoing run_{run_id}")
                tracker.update()
            elif not tracker.has_artifact():
                logger.debug(f"Saving artifact for completed run_{run_id}")
                artifact_success = tracker.save_artifact()
                if not artifact_success:
                    logger.debug(
                        f"Could not save artifact for run_{run_id}: Untracking run"
                    )
                    completed_runs.append(run_id)
            else:
                completed_runs.append(run_id)

        for completed_run_id in completed_runs:
            self.remove_run(completed_run_id)
