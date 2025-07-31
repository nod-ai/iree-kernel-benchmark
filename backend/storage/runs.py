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
from tqdm import tqdm
import os
from dotenv import load_dotenv
import time
import traceback


def update_runs(
    repo: Repository.Repository, db_client: DatabaseClient, dir_client: DirectoryClient
):
    print("Updating runs")
    stored_incomplete_runs = db_client.query_runs(
        (
            "status eq 'requested' or "
            "status eq 'in_progress' or "
            "status eq 'queued' or "
            "status eq 'pending'"
        )
    )
    if len(stored_incomplete_runs) == 0:
        print("No incomplete runs found")
        return
    else:
        print(f"Found {len(stored_incomplete_runs)} incomplete runs")

    for stored_run in stored_incomplete_runs:
        print(f"Loading run_{stored_run._id}")
        gh_run = repo.get_workflow_run(int(stored_run._id))
        gh_jobs = gh_run.jobs()

        for gh_job in gh_jobs:
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
    print("Updating artifacts")
    completed_runs = db_client.query_runs(
        ("status eq 'completed' and hasArtifact eq false")
    )
    if len(completed_runs) == 0:
        print("No completed runs without artifacts found")
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
    print("Updating change statistics")
    statless_runs = db_client.query_runs(
        f"status eq 'completed' and hasArtifact eq true and changeStats eq '{json.dumps({})}'"
    )
    if len(statless_runs) == 0:
        print("No statless runs with artifacts found")
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

    # validated_runs = db_client.query_runs(f"status eq 'completed' and hasArtifact eq true and changeStats ne '{json.dumps({})}'")
    # validated_runs = sorted(validated_runs, key=lambda run: run.timestamp, reverse=True)

    # for statless_run in statless_runs:
    #     print(f'Getting stats for run_{statless_run._id}')
    #     new_kernels = load_artifact_kernels(dir_client, f"{statless_run.blobName}/benchmark-results")
    #     change_stats = None

    #     for valid_run in validated_runs:
    #         if valid_run.headSha == statless_run.headSha:
    #             continue

    #         valid_merges = db_client.query_modifications(f"type eq 'merge' and headSha eq '{valid_run.headSha}'")
    #         if len(valid_merges) == 0:
    #             continue

    #         print(f'Found merge {valid_merges[0]['url']} for comparison')
    #         old_kernels = load_artifact_kernels(dir_client, f"{valid_run.blobName}/benchmark-results")
    #         change_stats = compare_artifact_kernels(old_kernels, new_kernels)

    #         break

    #     if not change_stats:
    #         print(f'Could not find comparison for run_{statless_run._id}. Saving no change')
    #         change_stats = compare_artifact_kernels(new_kernels)

    #     db_client.update_run(statless_run._id, { 'changeStats': change_stats })
