from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# @dataclass
# class RunResultEntry:
#     triggerId: str
#     blobName: str
#     runType: str
#     timestamp: datetime

type ChangeStats = dict[str, float]


@dataclass
class BenchmarkRun:
    _id: str
    headSha: str
    status: str
    conclusion: str
    numSteps: int
    steps: list[dict]
    blobName: str
    timestamp: datetime
    changeStats: ChangeStats
    hasArtifact: bool = False


@dataclass
class RunArtifact:
    kernels: list[dict]
    mapping: BenchmarkRun


@dataclass
class ChangeAuthor:
    name: str
    profileUrl: str


@dataclass
class RepoModification:
    _id: str
    headSha: str
    url: str
    type: str
    timestamp: datetime
    author: ChangeAuthor


@dataclass
class RepoCommit:
    _id: str
    title: str
    author: ChangeAuthor
    timestamp: datetime
    description: Optional[str] = None


@dataclass
class RepoPullRequest(RepoModification):
    title: str
    status: str
    commits: list[RepoCommit]
    description: Optional[str] = None


@dataclass
class RepoMerge(RepoModification):
    prId: str
