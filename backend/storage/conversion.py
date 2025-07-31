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


def convert_prs_from_github(pr_json: list[dict]) -> list[dict]:
    modifications = []

    for pr_dict in tqdm(pr_json, desc="Parsing PRs from JSON"):
        user_dict = pr_dict.get("user", {})

        author = ChangeAuthor(
            name=user_dict.get("name") or user_dict.get("login", "Anonymous"),
            profileUrl=user_dict.get("avatar_url"),
        )

        pr = RepoPullRequest(
            headSha=pr_dict["head"]["sha"],
            _id=str(pr_dict.get("id")),
            type="pr",
            timestamp=datetime.fromisoformat(pr_dict.get("created_at")),
            url=pr_dict.get("html_url"),
            author=author,
            title=pr_dict.get("title"),
            status=pr_dict.get("state"),
            commits=[],
            description=pr_dict.get("body"),
        )
        modifications.append(asdict(pr))

        if pr_dict.get("merged"):
            merge = RepoMerge(
                headSha=pr_dict["head"]["sha"],
                _id=str(pr_dict.get("id")) + "_merge",
                url=pr_dict.get("html_url"),
                type="merge",
                timestamp=datetime.fromisoformat(pr_dict.get("merged_at")),
                author=author,
                prId=str(pr_dict.get("id")),
            )
            modifications.append(asdict(merge))

    return modifications
