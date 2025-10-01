import logging
from backend.github_utils import get_repo
from backend.globals import BENCH_REPO_NAME
from backend.runs.run_utils import parse_run_from_json
from backend.runs.workflows import WorkflowRunInfo, find_workflow
from backend.storage.auth import get_blob_client
from backend.storage.types import *
from dataclass_wizard import asdict
import json

logger = logging.getLogger(__name__)


def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)


WORKFLOW_TO_RUN_TYPE = {
    "Short Benchmark": "bench",
    "Tune Wave Kernels": "tune",
}


class WorkflowListener:
    def __init__(self):
        self._repo = get_repo("bench")
        self._storage_client = get_blob_client()

    def _handle_workflow_run_requested(
        self, workflow_info: WorkflowRunInfo, run_payload: dict
    ):
        run_data = run_payload["workflow_run"]
        run = parse_run_from_json(run_data)
        logger.info(f"adding new run\n{jsonify(run)}")
        WorkflowRunDb.upsert(run)

    def _handle_workflow_run_progress(
        self, workflow_info: WorkflowRunInfo, run_payload: dict
    ):
        run_data = run_payload["workflow_run"]
        run_id = str(run_data["id"])
        run_type = workflow_info.run_type.name

        logger.info(f"updating run {run_id=} {run_type=} status={run_data['status']}")
        WorkflowRunDb.update_by_id(
            run_id,
            {
                "status": run_data["status"],
                "conclusion": run_data["conclusion"],
                "completed": run_data["status"] == "completed",
                "timestamp": datetime.fromisoformat(run_data["updated_at"]),
            },
        )

    def handle_workflow_run_payload(self, run_payload: dict):
        if run_payload["workflow_run"]["event"] != "workflow_dispatch":
            return
        if run_payload["repository"]["full_name"] != BENCH_REPO_NAME:
            return

        workflow_info = find_workflow(name=run_payload["workflow_run"]["name"])
        if not workflow_info:
            return

        handler = {
            "requested": self._handle_workflow_run_requested,
            "in_progress": self._handle_workflow_run_progress,
            "completed": self._handle_workflow_run_progress,
        }

        handler[run_payload["action"]](workflow_info, run_payload)

    def handle_workflow_job_payload(self, job_payload: dict):
        if job_payload["repository"]["full_name"] != BENCH_REPO_NAME:
            return
        if not find_workflow(main_job=job_payload["workflow_job"]["name"]):
            return

        run_id = str(job_payload["workflow_job"]["run_id"])
        steps = job_payload["workflow_job"]["steps"]

        logger.info("updating job", json.dumps(steps, indent=4))

        try:
            WorkflowRunDb.update_by_id(run_id, {"steps": steps})
        except Exception as e:
            return
