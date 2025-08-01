import json
from github import Github, Auth, PullRequest, Repository
from datetime import datetime, timezone
from globals import TESTING_MODE
from storage.artifacts import (
    load_artifact_kernels,
    save_run_artifact,
    compare_artifact_kernels,
)
from storage.db import DatabaseClient
from storage.directory import DirectoryClient


def update_run_mappings(
    repo: Repository.Repository, db_client: DatabaseClient, dir_client: DirectoryClient
):
    unmapped_runs = db_client.query_runs(
        "headSha eq 'undefined' and conclusion ne 'cancelled'"
    )

    if len(unmapped_runs) == 0:
        return
    else:
        print(f"Found {len(unmapped_runs)} unmapped runs")

    for run in unmapped_runs:
        gh_run = repo.get_workflow_run(int(run._id))
        gh_jobs = gh_run.jobs()

        head_sha = None

        for gh_job in gh_jobs:
            if "Identifier" not in gh_job.name:
                continue

            for job_step in gh_job.steps:
                if job_step.name.startswith("headSha_"):
                    head_sha = job_step.name.split("headSha_")[1]
                    break

        if head_sha:
            print(f"Successfully found identifier for run_{run._id} -- {head_sha}")
            db_client.update_run(run._id, {"headSha": head_sha})
        else:
            print(f"Failed to find identifier for run_{run._id}")


def update_runs(
    repo: Repository.Repository, db_client: DatabaseClient, dir_client: DirectoryClient
):
    stored_incomplete_runs = db_client.query_runs(
        (
            "status eq 'requested' or "
            "status eq 'in_progress' or "
            "status eq 'queued' or "
            "status eq 'pending'"
        )
    )
    if len(stored_incomplete_runs) == 0:
        return
    else:
        print(f"Found {len(stored_incomplete_runs)} incomplete runs")

    for stored_run in stored_incomplete_runs:
        print(f"Loading run_{stored_run._id}")
        gh_run = repo.get_workflow_run(int(stored_run._id))
        gh_jobs = gh_run.jobs()

        for gh_job in gh_jobs:
            if "Short Benchmark" not in gh_job.name:
                continue

            job_steps = [
                {
                    "name": step.name,
                    "status": step.status,
                    "conclusion": step.conclusion,
                    "number": step.number,
                    "started_at": (
                        datetime.isoformat(step.started_at) if step.started_at else None
                    ),
                    "completed_at": (
                        datetime.isoformat(step.completed_at)
                        if step.completed_at
                        else None
                    ),
                }
                for step in gh_job.steps
            ]

            print(f"Updating run_{stored_run._id} in db")
            db_client.update_run(
                stored_run._id,
                {
                    "status": gh_run.status,
                    "conclusion": gh_run.conclusion,
                    "numSteps": len(job_steps),
                    "steps": job_steps,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            break


def update_artifacts(
    repo: Repository.Repository, db_client: DatabaseClient, dir_client: DirectoryClient
):
    completed_runs = db_client.query_runs(
        ("status eq 'completed' and conclusion eq 'success' and hasArtifact eq false")
    )
    if len(completed_runs) == 0:
        return

    for completed_run in completed_runs:
        print(f"Saving run_{completed_run._id} artifact")
        save_success = save_run_artifact(repo, completed_run, dir_client)

        if save_success:
            db_client.update_run(
                completed_run._id,
                {
                    "hasArtifact": True,
                },
            )


def update_change_stats(
    repo: Repository.Repository, db_client: DatabaseClient, dir_client: DirectoryClient
):
    statless_runs = db_client.query_runs(
        f"status eq 'completed' and hasArtifact eq true and changeStats eq '{json.dumps({})}'"
    )
    if len(statless_runs) == 0:
        return

    for statless_run in statless_runs:
        print(f"Getting stats for run_{statless_run._id}")
        baseline_kernels = load_artifact_kernels(
            dir_client, "baseline/benchmark-results"
        )
        new_kernels = load_artifact_kernels(
            dir_client, f"{statless_run.blobName}/benchmark-results"
        )
        change_stats = compare_artifact_kernels(baseline_kernels, new_kernels)
        db_client.update_run(statless_run._id, {"changeStats": change_stats})
