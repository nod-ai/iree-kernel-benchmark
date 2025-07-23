from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RunResultEntry:
    triggerId: str
    blobName: str
    runType: str
    timestamp: datetime

@dataclass
class RunArtifact:
    kernels: list[dict]
    mapping: RunResultEntry

@dataclass
class ChangeAuthor:
    name: str
    profileUrl: str

@dataclass
class RepoModification:
    _id: str
    url: str
    type: str
    timestamp: datetime
    author: ChangeAuthor
    changeStats: dict[str, float]

@dataclass
class RepoCommit:
    _id: str
    title: str
    author: ChangeAuthor
    timestamp: datetime
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