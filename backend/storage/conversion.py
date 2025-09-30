from typing import Any, Dict, Tuple
from .types import *
from datetime import datetime
import random
from dataclass_wizard import asdict
from tqdm import tqdm


def random_stats() -> dict[str, float]:
    KERNEL_TYPES = ["gemm", "attention", "convolution"]
    stats = {}
    for k in KERNEL_TYPES:
        # Random float between -50 and +100, rounded to 2 decimal places
        stats[k] = round(random.uniform(-50, 100), 2)
    return stats


def parse_pr_obj(pr_dict: Dict[str, Any]) -> RepoPullRequest:
    user_dict: dict = pr_dict.get("user", {})

    author = ChangeAuthor(
        name=user_dict.get("name") or user_dict.get("login", "Anonymous"),
        profileUrl=user_dict.get("avatar_url"),
    )

    pr = RepoPullRequest(
        _id=str(pr_dict.get("id")),
        url=pr_dict.get("html_url"),
        type="pr",
        timestamp=datetime.fromisoformat(pr_dict.get("created_at")),
        author=author,
        title=pr_dict.get("title"),
        status=pr_dict.get("state"),
        commits=pr_dict.get("commits"),
        repoName=pr_dict["head"]["repo"]["full_name"],
        branchName=pr_dict["head"]["ref"],
        mappingId=pr_dict["head"]["sha"],
        description=pr_dict.get("body"),
        isMerged=pr_dict.get("merged"),
    )

    return pr


def convert_prs_from_github(pr_json: list[dict]) -> list[dict]:
    modifications = []

    for pr_dict in tqdm(pr_json, desc="Parsing PRs from JSON"):
        pr = parse_pr_obj(pr_dict)
        modifications.append(asdict(pr))

    return modifications
