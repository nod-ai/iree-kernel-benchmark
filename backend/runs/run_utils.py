from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4
from dataclass_wizard import fromdict
from backend.runs.workflows import find_workflow
from backend.storage.auth import get_blob_client
from backend.storage.types import (
    WorkflowRunDb,
    WorkflowRunState,
)
from github.WorkflowRun import WorkflowRun as GhWorkflowRun


RUN_INCOMPLETE_STATUSES = ["requested", "in_progress", "queued", "pending"]


def get_run_by_blob_name(blob_name: str) -> WorkflowRunState:
    runs = WorkflowRunDb.query(f"blobName eq '{blob_name}'")
    return runs[0]


def parse_run_from_gh(gh_run: GhWorkflowRun) -> WorkflowRunState:
    workflow = find_workflow(name=gh_run.name)
    if not workflow:
        raise ValueError(f"Workflow {gh_run.name} is unsupported")

    steps = []
    if workflow.main_job:
        for gh_job in gh_run.jobs():
            if gh_job.name == workflow.main_job:
                steps = gh_job.raw_data["steps"]
                break

    return WorkflowRunState(
        _id=str(gh_run.id),
        type=workflow.run_type.name,
        blobName=str(gh_run.id),
        timestamp=gh_run.updated_at,
        status=gh_run.status,
        conclusion=gh_run.conclusion,
        numSteps=10,
        steps=steps,
        completed=gh_run.completed,
        hasArtifact=False,
        mappingId=None,
    )


def parse_run_from_json(run_json: dict[str, Any]) -> WorkflowRunState:
    workflow = find_workflow(name=run_json["name"])
    if not workflow:
        raise ValueError(f"Workflow {run_json['name']} is unsupported")

    run_id = str(run_json["id"])
    run_type = workflow.run_type.name

    return WorkflowRunState(
        _id=run_id,
        type=run_type,
        blobName=str(uuid4()),
        timestamp=datetime.fromisoformat(run_json["updated_at"]),
        status=run_json["status"],
        conclusion=run_json["conclusion"] or "unknown",
        numSteps=10,
        steps=[],
        completed=False,
        hasArtifact=False,
        mappingId="undefined",
    )


def find_incomplete_runs() -> List[WorkflowRunState]:
    incomplete_runs = WorkflowRunDb.query(
        " or ".join([f"status eq '{status}'" for status in RUN_INCOMPLETE_STATUSES])
    )
    artifactless_runs = WorkflowRunDb.query(
        f"conclusion eq 'success' and hasArtifact eq false"
    )
    return incomplete_runs + artifactless_runs
