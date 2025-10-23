from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, List, Optional

from dataclass_wizard import asdict

from backend.github_utils.actions import trigger_workflow_dispatch
from backend.github_utils.gist import create_gist
from backend.globals import BENCH_REPO_BRANCH
from backend.runs import RunType
from backend.storage.types import (
    KernelConfig,
    KernelConfigDb,
    TuningConfig,
    TuningConfigDb,
)

logger = logging.getLogger(__name__)


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
        identifier="tuningRunId",
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


def trigger_short_bench_run(
    machine: str = "mi325",
    problems: Optional[List[KernelConfig]] = None,
    tuned_configs: Optional[List[TuningConfig]] = None,
    **kwargs,
) -> bool:
    if problems:
        logger.info(f"Using {len(problems)} custom kernels for quick benchmark")
    else:
        problems = KernelConfigDb.find_all({"workflow": "all"})
        logger.info(f"Loaded {len(problems)} default kernels for quick benchmark")

    problems_json = [asdict(p) for p in problems]
    problems_gist = create_gist(problems_json)

    if tuned_configs:
        logger.info(
            f"Using {len(tuned_configs)} custom tuning configurations for quick benchmark"
        )
    else:
        tuned_configs = find_latest_tuned_configs(problems)
        logger.info(
            f"Loaded {len(tuned_configs)} default tuning configurations for quick benchmark"
        )
    tuned_configs_json = [asdict(c) for c in tuned_configs]
    tuned_configs_gist = create_gist(tuned_configs_json)

    inputs = {
        "machine": machine,
        "problems_url": problems_gist.raw_url,
        "tuned_config_url": tuned_configs_gist.raw_url,
        **kwargs,
    }

    return trigger_bench_workflow(RunType.BENCHMARK, inputs)


def find_latest_tuned_configs(problems: List[KernelConfig]) -> List[TuningConfig]:
    problem_configs = {p.name: None for p in problems}

    all_configs = TuningConfigDb.find_all()
    for config in all_configs:
        name = config.kernel_name
        if name not in problem_configs:
            continue

        if (
            problem_configs[name] is None
            or config.timestamp > problem_configs[name].timestamp
        ):
            problem_configs[name] = config

    return [config for config in problem_configs.values() if config]
