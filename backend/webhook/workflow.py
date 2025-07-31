from auth import get_repo
from storage.conversion import random_stats
from globals import TESTING_MODE
from storage.db import DatabaseClient
from storage.directory import DirectoryClient
from storage.types import *
from storage.artifacts import download_artifact_kernels_by_run_id, save_run_artifact
from dataclass_wizard import asdict
from github import Github, Auth
import json
from azure.core.exceptions import ResourceNotFoundError


def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)


class WorkflowListener:
    def __init__(self, db_client: DatabaseClient, storage_client: DirectoryClient):
        self._repo = get_repo("bench")
        self._db_client = db_client
        self._storage_client = storage_client

    def _handle_workflow_run_requested(self, run_payload: dict):
        run_data = run_payload["workflow_run"]
        run_id = str(run_data["id"])

        pr = self._db_client.find_latest_pr()

        run = BenchmarkRun(
            _id=run_id,
            headSha=pr.headSha,
            status=run_data["status"],
            conclusion=run_data["conclusion"] or "unknown",
            numSteps=7 if TESTING_MODE else 17,
            steps=[],
            blobName=run_data["updated_at"],
            timestamp=datetime.fromisoformat(run_data["updated_at"]),
            changeStats={},
            hasArtifact=False,
        )
        print("adding new run", run_id, jsonify(run))
        self._db_client.insert_run(run)

    def _handle_workflow_run_progress(self, run_payload: dict):
        run_data = run_payload["workflow_run"]

        run_id = str(run_data["id"])

        print(
            "updating run",
            run_id,
            json.dumps(
                {
                    "status": run_data["status"],
                    "conclusion": run_data["conclusion"] or "unknown",
                    "updated_at": run_data["updated_at"],
                },
                indent=4,
            ),
        )

        self._db_client.update_run(
            run_id,
            {
                "status": run_data["status"],
                "conclusion": run_data["conclusion"],
                "timestamp": datetime.fromisoformat(run_data["updated_at"]),
            },
        )

    def _handle_workflow_run_complete(self, run_payload: dict):
        run_data = run_payload["workflow_run"]
        run_id = str(run_data["id"])

        print(
            "completing run",
            run_id,
            json.dumps(
                {
                    "status": run_data["status"],
                    "conclusion": run_data["conclusion"] or "unknown",
                    "updated_at": run_data["updated_at"],
                },
                indent=4,
            ),
        )

        self._db_client.update_run(
            run_id,
            {
                "status": run_data["status"],
                "conclusion": run_data["conclusion"],
                "timestamp": datetime.fromisoformat(run_data["updated_at"]),
            },
        )

    def handle_workflow_run_payload(self, run_payload: dict):
        if run_payload["workflow_run"]["event"] != "workflow_dispatch":
            return
        if run_payload["repository"]["name"] != "iree-kernel-benchmark":
            return
        if run_payload["workflow_run"]["name"] != "Short Benchmark":
            return

        handler = {
            "requested": self._handle_workflow_run_requested,
            "in_progress": self._handle_workflow_run_progress,
            "completed": self._handle_workflow_run_complete,
        }

        handler[run_payload["action"]](run_payload)

    def handle_workflow_job_payload(self, job_payload: dict):
        if job_payload["repository"]["name"] != "iree-kernel-benchmark":
            return
        if job_payload["workflow_job"]["name"] != "short_benchmark":
            return

        run_id = str(job_payload["workflow_job"]["run_id"])
        steps = job_payload["workflow_job"]["steps"]

        print("updating job", json.dumps(steps, indent=4))

        try:
            self._db_client.update_run(run_id, {"steps": steps})
        except Exception as e:
            return
