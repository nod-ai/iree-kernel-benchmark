from dataclasses import dataclass
from typing import Optional

@dataclass
class RunResultEntry:
    triggerId: str
    blobName: str
    runType: str
    timestamp: int

@dataclass
class ChangeAuthor:
    name: str
    profileUrl: str

@dataclass
class RepoModification:
    _id: str
    url: str
    type: str
    timestamp: int
    author: ChangeAuthor
    changeStats: dict[str, float]

@dataclass
class RepoCommit:
    _id: str
    title: str
    author: ChangeAuthor
    timestamp: int
    description: Optional[str]

@dataclass
class RepoPullRequest(RepoModification):
    title: str
    description: Optional[str]
    status: str
    commits: list[RepoCommit]

@dataclass
class RepoMerge(RepoModification):
    prId: str