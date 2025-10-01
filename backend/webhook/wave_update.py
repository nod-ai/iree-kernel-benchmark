import logging
from backend.github_utils import get_repo, get_github_token
from backend.github_utils.actions import trigger_workflow_dispatch
from backend.globals import (
    BENCH_ITERATIONS,
    BENCH_REPO_BRANCH,
    MAX_BENCH_KERNELS,
    RUN_ALL_BACKENDS,
    WAVE_REPO_NAME,
)
from backend.runs import RunType
from backend.runs.workflows import find_workflow
from backend.storage.auth import get_blob_client
from backend.storage.conversion import parse_pr_obj
from backend.storage.directory import DirectoryClient
from backend.storage.types import *
from datetime import timezone
from dataclass_wizard import asdict
import json
import requests

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self._wave_repo = get_repo("wave")
        self._bench_repo = get_repo("bench")
        self._storage_client = get_blob_client()

    def trigger_workflow(
        self, repo_name: str, branch_name: str, head_sha: str, metadata: dict = None
    ) -> bool:
        inputs = {
            "max_kernels": MAX_BENCH_KERNELS,
            "selected_backend": "all" if RUN_ALL_BACKENDS else "wave",
            "selected_kernel": "all",
            "pr_repository": repo_name,
            "pr_branch": branch_name,
            "pr_headsha": head_sha,
        }
        if metadata:
            inputs["metadata"] = json.dumps(metadata)

        return trigger_workflow_dispatch(
            repo_id="bench",
            branch_name=BENCH_REPO_BRANCH,
            workflow_id=find_workflow(RunType.BENCHMARK).filename,
            inputs=inputs,
        )

    def handle_pr_payload(self, pr_payload: dict):
        action = pr_payload["action"]
        pr_obj = pr_payload["pull_request"]

        # if pr_obj["draft"]:
        #     return

        # ignore if not being merged into iree-org/wave/main
        if (
            pr_obj["base"]["repo"]["full_name"] != WAVE_REPO_NAME
            or pr_obj["base"]["ref"] != "main"
        ):
            return

        is_merge = pr_obj["merged"]
        entry_id = str(pr_obj["id"])

        try:
            mod = RepoPullRequestDb.find_by_id(entry_id)
            logger.debug("Found modification in database")
            has_changed = pr_obj["commits"] != mod.commits
        except:
            logger.debug("Modification not found in database")
            has_changed = True

        pr = parse_pr_obj(pr_obj)
        RepoPullRequestDb.upsert(pr)

        if has_changed and not is_merge:
            logger.info(
                f'Pull Request {pr_obj["html_url"]} triggering workflow on {action}'
            )
            head_repo_name = pr_obj["head"]["repo"]["full_name"]
            head_branch = pr_obj["head"]["ref"]
            head_sha = pr_obj["head"]["sha"]
            self.trigger_workflow(
                head_repo_name, head_branch, head_sha
            )  # { 'trigger': pr_entry }
