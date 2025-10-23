import logging
from backend.github_utils import get_repo
from backend.globals import WAVE_REPO_NAME
from backend.runs import RunType
from backend.runs.workflows import trigger_short_bench_run
from backend.storage.auth import get_blob_client
from backend.storage.conversion import parse_pr_obj
from backend.storage.types import *
from dataclass_wizard import asdict
import json

logger = logging.getLogger(__name__)


def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)


class WaveUpdateListener:
    def __init__(self):
        self._wave_repo = get_repo("wave")
        self._bench_repo = get_repo("bench")
        self._storage_client = get_blob_client()

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

        if not has_changed or is_merge:
            return

        logger.info(
            f'Pull Request {pr_obj["html_url"]} triggering workflow on {action}'
        )
        head_repo_name = pr_obj["head"]["repo"]["full_name"]
        head_branch = pr_obj["head"]["ref"]
        head_sha = pr_obj["head"]["sha"]
        trigger_success = trigger_short_bench_run(
            pr_repository=head_repo_name, pr_branch=head_branch, pr_headsha=head_sha
        )
        if trigger_success:
            logger.info(
                f'Successfully triggered workflow for Pull Request {pr_obj["html_url"]}'
            )
        else:
            logger.error(
                f'Failed to trigger workflow for Pull Request {pr_obj["html_url"]}'
            )
