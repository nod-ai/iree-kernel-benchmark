from typing import Any, Dict
from .types import *
from datetime import datetime


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
