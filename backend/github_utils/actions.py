from dataclasses import is_dataclass
import json

from dataclass_wizard import asdict
from .auth import get_github_token, get_repo
import requests
from typing import Any


def trigger_workflow_dispatch(
    repo_id: str, branch_name: str, workflow_id: str, inputs: dict[str, Any]
) -> bool:
    def stringify_value(val: Any):
        if is_dataclass(val):
            return json.dumps(asdict(val))
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        return str(val)

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
        "inputs": {key: stringify_value(val) for key, val in inputs.items()},
    }

    print(
        f"Triggering workflow {repo_name=} {branch_name=} {workflow_id=}:\n{json.dumps(inputs, indent=4)}"
    )

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 204:
        print(f"Workflow failed: {response.json()}")
        return False
    print(f"Workflow dispatched successfully: {response.status_code}")
    return True
