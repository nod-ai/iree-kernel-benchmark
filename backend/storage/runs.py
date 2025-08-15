import json
import os
from typing import Any
from github import Github, Auth, PullRequest, Repository
from datetime import datetime, timezone
import requests
from globals import TESTING_MODE
from auth import get_azure_clients, get_gist_token, get_repo, get_github_token
from storage.artifacts import (
    load_artifact_kernels,
    save_run_artifact,
    compare_artifact_kernels,
    save_tuning_run_artifact,
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


def update_tuning_runs():
    db_client, dir_client = get_azure_clients()
    runs = db_client.query_tuning_runs(("hasArtifact eq false and completed eq true"))
    for run in runs:
        print(f"Saving run artifact for run {run._id}")
        save_tuning_run_artifact(run)
        db_client.update_tuning_run(run._id, {"hasArtifact": True})

        updated_configs = db_client.find_latest_tuning_configs()
        update_json_gist(os.getenv("TUNING_GIST_ID"), updated_configs)


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


def trigger_workflow_dispatch(
    repo_id: str, branch_name: str, workflow_id: str, inputs: dict[str, Any]
) -> bool:
    token = get_github_token(repo_id)
    repo = get_repo(repo_id)
    repo_name = repo.full_name

    url = f"https://api.github.com/repos/{repo_name}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {
        "ref": branch_name,
        "inputs": inputs,
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 204:
        print(f"Workflow failed: {response.json()}")
        return False
    print(f"Workflow dispatched successfully: {response.status_code}")
    return True


def upload_json_to_gist(data, filename=None, description=None):
    github_token = get_gist_token()

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    json_content = json.dumps(data, indent=2)

    gist_data = {
        "description": description or "Temporary JSON storage",
        "public": True,
        "files": {filename: {"content": json_content}},
    }

    headers = {"Accept": "application/vnd.github.v3+json"}

    headers["Authorization"] = f"token {github_token}"

    response = requests.post(
        "https://api.github.com/gists", headers=headers, json=gist_data
    )

    if response.status_code == 201:
        gist_info = response.json()
        raw_url = gist_info["files"][filename]["raw_url"]

        return {
            "raw_url": raw_url,
            "gist_url": gist_info["html_url"],
            "gist_id": gist_info["id"],
            "success": True,
        }
    else:
        return {
            "success": False,
            "error": f"Failed to create gist: {response.status_code} - {response.text}",
        }


def update_json_gist(gist_id, data, filename=None, description=None):
    github_token = get_gist_token()

    # First, get the existing gist to find the filename if not provided
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }

    # Get existing gist details
    get_response = requests.get(
        f"https://api.github.com/gists/{gist_id}", headers=headers
    )

    if get_response.status_code != 200:
        return {
            "success": False,
            "error": f"Failed to retrieve gist: {get_response.status_code} - {get_response.text}",
        }

    existing_gist = get_response.json()

    # If filename not provided, use the first JSON file from the gist
    if filename is None:
        json_files = [f for f in existing_gist["files"].keys() if f.endswith(".json")]
        if json_files:
            filename = json_files[0]
        else:
            # If no JSON files exist, create a new filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    json_content = json.dumps(data, indent=2)

    # Prepare update data
    update_data = {"files": {filename: {"content": json_content}}}

    # Add description if provided
    if description is not None:
        update_data["description"] = description

    # Update the gist
    response = requests.patch(
        f"https://api.github.com/gists/{gist_id}", headers=headers, json=update_data
    )

    if response.status_code == 200:
        gist_info = response.json()
        raw_url = gist_info["files"][filename]["raw_url"]

        return {
            "raw_url": raw_url,
            "gist_url": gist_info["html_url"],
            "gist_id": gist_info["id"],
            "filename": filename,
            "success": True,
        }
    else:
        return {
            "success": False,
            "error": f"Failed to update gist: {response.status_code} - {response.text}",
        }


def load_json_from_gist_url(raw_url: str):
    response = requests.get(raw_url)
    return response.json()
