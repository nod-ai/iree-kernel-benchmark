from backend.github_utils import get_repo, get_github_token
from backend.globals import BENCH_ITERATIONS, MAX_BENCH_KERNELS, RUN_ALL_BACKENDS
from backend.storage.db import DatabaseClient
from backend.storage.directory import DirectoryClient
from backend.storage.types import *
from datetime import timezone
from dataclass_wizard import asdict
import json
import requests


def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)


class WaveUpdateListener:
    """
    Wave Update Events:
    - assigned
    - auto_merge_disabled
    - auto_merge_enabled
    - closed
    - converted_to_draft
    - demilestoned
    - dequeued
    - edited
    - enqueued
    - labeled
    - locked
    - milestoned
    - opened
    - ready_for_review
    - reopened
    - review_request_removed
    - review_requested
    - synchronize
    - unassigned
    - unlabeled
    - unlocked
    """

    def __init__(self, db_client: DatabaseClient, storage_client: DirectoryClient):
        self._wave_repo = get_repo("wave")
        self._bench_repo = get_repo("bench")
        self._db_client = db_client
        self._storage_client = storage_client

    def trigger_workflow(
        self, repo_name: str, branch_name: str, head_sha: str, metadata: dict = None
    ) -> bool:
        bench_repo_name = self._bench_repo.full_name
        workflow_id = "short_bench.yml"
        token = get_github_token("bench")

        url = f"https://api.github.com/repos/{bench_repo_name}/actions/workflows/{workflow_id}/dispatches"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        body = {
            "ref": "kernel-dashboard",
            "inputs": {
                "iterations": str(BENCH_ITERATIONS),
                "max_kernels": str(MAX_BENCH_KERNELS),
                "selected_backend": "all" if RUN_ALL_BACKENDS else "wave",
                "pr_repository": repo_name,
                "pr_branch": branch_name,
                "pr_headsha": head_sha,
            },
        }
        if metadata:
            body["inputs"]["metadata"] = json.dumps(metadata)

        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 204:
            print(f"Workflow failed: {response.json()}")
            return False
        print(f"Workflow dispatched successfully: {response.status_code}")
        return True

    def _save_pr(self, id: str, pr_obj: dict) -> dict:
        author = ChangeAuthor(
            name=pr_obj["user"].get("name") or pr_obj["user"].get("login", "Anonymous"),
            profileUrl=pr_obj["user"]["avatar_url"],
        )
        pr = RepoPullRequest(
            _id=id,
            headSha=pr_obj["head"]["sha"],
            url=pr_obj["html_url"],
            type="pr",
            timestamp=datetime.now(
                timezone.utc
            ),  # fromisoformat(pr_obj['created_at']),
            author=author,
            title=pr_obj["title"],
            status=pr_obj["state"],
            commits=[],
            description=pr_obj["body"],
            repoName=pr_obj["head"]["repo"]["full_name"],
            branchName=pr_obj["head"]["ref"],
        )
        self._db_client.insert_pull_request(pr)

        if pr_obj["merged"]:
            merge = RepoMerge(
                _id=id,
                headSha=pr_obj["head"]["sha"],
                url=pr_obj["html_url"],
                type="merge",
                timestamp=datetime.fromisoformat(pr_obj["merged_at"]),
                author=author,
                prId=str(pr_obj["id"]),
            )
            self._db_client.insert_merge(merge)
            return asdict(merge)
        else:
            return asdict(pr)

    def handle_pr_payload(self, pr_payload: dict):
        action = pr_payload["action"]
        pr_obj = pr_payload["pull_request"]

        if pr_obj["draft"]:
            return

        # ignore if not being merged into iree-org/wave/main
        if (
            pr_obj["base"]["repo"]["full_name"] != "iree-org/wave"
            or pr_obj["base"]["ref"] != "main"
        ):
            return

        is_merge = pr_obj["merged"]
        entry_id = f"{pr_obj['id']}_merge" if is_merge else str(pr_obj["id"])

        try:
            mod = self._db_client.find_modification_by_id(
                "merge" if is_merge else "pr", entry_id
            )
            print("Found modification in database")
            has_changed = pr_obj["commits"] != mod.commits
        except:
            print("Modification not found in database")
            has_changed = True

        pr_entry = self._save_pr(entry_id, pr_obj)

        if has_changed and not is_merge:
            print(f'Pull Request {pr_obj["html_url"]} triggering workflow on {action}')
            head_repo_name = pr_obj["head"]["repo"]["full_name"]
            head_branch = pr_obj["head"]["ref"]
            head_sha = pr_obj["head"]["sha"]
            self.trigger_workflow(
                head_repo_name, head_branch, head_sha
            )  # { 'trigger': pr_entry }
