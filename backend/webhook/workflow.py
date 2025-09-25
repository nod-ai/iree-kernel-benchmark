import logging
from backend.github_utils import get_repo
from backend.storage.auth import get_azure_clients
from backend.storage.db import DatabaseClient
from backend.storage.directory import DirectoryClient
from backend.storage.types import *
from dataclass_wizard import asdict
import json


def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)


class WorkflowListener:
    def __init__(self):
        self._repo = get_repo("bench")
        self._db_client, self._storage_client = get_azure_clients()
        self._logger = logging.getLogger("backend")

    def _handle_workflow_run_requested(self, workflow_name: str, run_payload: dict):
        run_data = run_payload["workflow_run"]
        run_id = str(run_data["id"])

        if workflow_name == "Short Benchmark":
            run = BenchmarkRun(
                _id=run_id,
                headSha="undefined",
                status=run_data["status"],
                conclusion=run_data["conclusion"] or "unknown",
                numSteps=10,
                steps=[],
                blobName=run_data["updated_at"],
                timestamp=datetime.fromisoformat(run_data["updated_at"]),
                changeStats={},
                hasArtifact=False,
            )
            self._logger.info("adding new benchmark run", run_id, jsonify(run))
            self._db_client.insert_run(run)

        elif workflow_name == "Tune Wave Kernels":
            run = TuningRun(
                _id=run_id,
                blobName=run_id,
                timestamp=datetime.fromisoformat(run_data["updated_at"]),
                changeStats={},
                hasArtifact=False,
            )
            self._logger.info("adding new tuning run", run_id, jsonify(run))
            self._db_client.insert_tuning_run(run)

    def _handle_workflow_run_progress(self, workflow_name: str, run_payload: dict):
        run_data = run_payload["workflow_run"]
        run_id = str(run_data["id"])

        if workflow_name == "Short Benchmark":
            self._logger.info("updating benchmark run", run_id, run_data["status"])
            self._db_client.update_run(
                run_id,
                {
                    "status": run_data["status"],
                    "conclusion": run_data["conclusion"],
                    "timestamp": datetime.fromisoformat(run_data["updated_at"]),
                },
            )

        elif workflow_name == "Tune Wave Kernels":
            self._logger.info("updating tuning run", run_id, run_data["status"])

            if run_data["status"] == "completed":
                if run_data["conclusion"] == "success":
                    self._db_client.update_tuning_run(run_id, {"completed": True})
                else:
                    self._db_client.delete_tuning_run(run_id)

    def handle_workflow_run_payload(self, run_payload: dict):
        if run_payload["workflow_run"]["event"] != "workflow_dispatch":
            return
        if run_payload["repository"]["name"] != "iree-kernel-benchmark":
            return

        workflow_name = run_payload["workflow_run"]["name"]

        handler = {
            "requested": self._handle_workflow_run_requested,
            "in_progress": self._handle_workflow_run_progress,
            "completed": self._handle_workflow_run_progress,
        }

        handler[run_payload["action"]](workflow_name, run_payload)

    def handle_workflow_job_payload(self, job_payload: dict):
        if job_payload["repository"]["name"] != "iree-kernel-benchmark":
            return
        if job_payload["workflow_job"]["name"] != "short_benchmark":
            return

        run_id = str(job_payload["workflow_job"]["run_id"])
        steps = job_payload["workflow_job"]["steps"]

        self._logger.info("updating job", json.dumps(steps, indent=4))

        try:
            self._db_client.update_run(run_id, {"steps": steps})
        except Exception as e:
            return
