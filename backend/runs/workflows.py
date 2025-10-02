from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from backend.github_utils.actions import trigger_workflow_dispatch
from backend.globals import BENCH_REPO_BRANCH
from backend.runs import RunType


@dataclass
class WorkflowRunInfo:
    run_type: RunType
    name: str
    filename: str
    main_job: Optional[str] = None
    identifier: Optional[str] = None


SUPPORTED_WORKFLOWS = [
    WorkflowRunInfo(
        run_type=RunType.BENCHMARK,
        name="Short Benchmark",
        filename="short_bench.yml",
        main_job="Short Benchmark",
        identifier="headSha",
    ),
    WorkflowRunInfo(
        run_type=RunType.TUNING,
        name="Tune Wave Kernels",
        filename="tune_kernels.yml",
        main_job="Tune Kernels",
    ),
    WorkflowRunInfo(
        run_type=RunType.E2E,
        name="Performance",
        filename="run_bench.yml",
        main_job="E2E Benchmarks",
    ),
]


def find_workflow(
    run_type: str | RunType = None,
    name: str = None,
    filename: str = None,
    main_job: str = None,
) -> Optional[WorkflowRunInfo]:
    for workflow_info in SUPPORTED_WORKFLOWS:
        if run_type:
            if isinstance(run_type, str):
                if RunType[run_type] != workflow_info.run_type:
                    continue
            elif isinstance(run_type, RunType):
                if run_type != workflow_info.run_type:
                    continue

        if name and name != workflow_info.name:
            continue
        if filename and filename != workflow_info.filename:
            continue
        if main_job and main_job != workflow_info.main_job:
            continue

        return workflow_info
    return None


def trigger_bench_workflow(
    run_type: RunType,
    inputs: Optional[dict[str, Any]] = None,
) -> bool:
    workflow = find_workflow(run_type)
    if not workflow:
        return False

    return trigger_workflow_dispatch(
        repo_id="bench",
        branch_name=BENCH_REPO_BRANCH,
        workflow_id=workflow.filename,
        inputs=inputs,
    )
