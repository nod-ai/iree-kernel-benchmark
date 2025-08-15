from auth import get_azure_clients, get_repo
from .conversion import parse_modification
from .artifacts import (
    download_artifact_kernels,
    compare_artifact_kernels,
    save_results_from_local_path,
)
from .types import *
from pathlib import Path
import shutil
import json
from dataclass_wizard import asdict


def rebase_performance(limit=10):
    db_client, dir_client = get_azure_clients()
    wave = get_repo("bench")

    gh_perf_runs = wave.get_workflow("run_bench.yml").get_runs(status="completed")

    perf_runs: list[PerformanceRun] = []
    perf_kernels: list[tuple[list[dict], Path]] = []

    i = 0
    for run in gh_perf_runs:
        for artifact in run.get_artifacts():
            if artifact.name != "benchmark-results":
                continue

            try:
                artifact_kernels, kernel_path = download_artifact_kernels(artifact)
            except:
                break

            date_str = run.created_at.strftime("%Y-%m-%d %H:%M:%S")

            perf_run = PerformanceRun(
                _id=str(run.id),
                blobName=str(run.id),
                timestamp=run.created_at,
                changeStats={},
            )

            perf_runs.append(perf_run)
            perf_kernels.append((artifact_kernels, kernel_path))

            print(
                f"Run {run.id} created at {date_str} with {len(artifact_kernels)} kernels"
            )

            break

        i += 1
        if i >= limit:
            break

    if len(perf_runs) == 0 or len(perf_kernels) == 0:
        return []

    old_kernels = perf_kernels[-1][0]
    newest_path = perf_kernels[0][1]

    dir_client.rm("baseline", recursive=True)
    save_results_from_local_path(
        dir_client, newest_path, "baseline", delete_local=False
    )

    for i, (new_kernels, kernel_path) in enumerate(perf_kernels):
        perf_run = perf_runs[i]
        try:
            perf_run.changeStats = compare_artifact_kernels(old_kernels, new_kernels)
        except:
            perf_run.changeStats = {}

        db_client.insert_performance(perf_run)
        save_results_from_local_path(dir_client, kernel_path, perf_run.blobName)

    db_performances = db_client.find_all_performances()

    performance_dicts = [asdict(db_perf) for db_perf in db_performances]
    # print(json.dumps(performance_dicts, indent=4))
    return performance_dicts


def rebase_modifications(limit=40):
    db_client, dir_client = get_azure_clients()
    db_client.clear_all_repos()

    wave_repo = get_repo("wave")

    open_prs = wave_repo.get_pulls(state="all", sort="created", direction="desc")

    i = 0
    for gh_pr in open_prs:
        pr_dict = gh_pr.raw_data
        pr, merge = parse_modification(pr_dict)

        db_client.insert_pull_request(pr)
        if merge:
            db_client.insert_merge(merge)

        i += 1
        if i >= limit:
            return


def rebase_all(mod_limit=40, perf_limit=20):
    rebase_modifications(mod_limit)
    rebase_performance(perf_limit)
