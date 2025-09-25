from .auth import get_github_token, get_repo
import requests
from typing import Any


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
