import json
from typing import Any, Dict, List
from dataclass_wizard import fromdict
from backend.storage.auth import get_blob_client
from backend.storage.types import (
    BenchmarkRun,
    BenchmarkRunDb,
    TuningRun,
    WorkflowRunBase,
)


RUN_INCOMPLETE_STATUSES = ["requested", "in_progress", "queued", "pending"]


def get_run_by_blob_name(blob_name: str) -> WorkflowRunBase:
    runs = BenchmarkRunDb.query(f"blobName eq '{blob_name}'")
    return runs[0]


def parse_run_json(run_json: Dict[str, Any]) -> WorkflowRunBase:
    run_json["steps"] = json.loads(run_json["steps"])
    if "changeStats" in run_json:
        run_json["changeStats"] = json.loads(run_json["changeStats"])
        return fromdict(BenchmarkRun, run_json)
    else:
        return fromdict(TuningRun, run_json)


def find_incomplete_runs() -> List[WorkflowRunBase]:
    run_jsons = BenchmarkRunDb.query(
        " or ".join([f"status eq {status}" for status in RUN_INCOMPLETE_STATUSES])
    )
    return list(map(parse_run_json, run_jsons))
