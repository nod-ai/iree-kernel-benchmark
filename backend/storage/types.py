from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

# @dataclass
# class RunResultEntry:
#     triggerId: str
#     blobName: str
#     runType: str
#     timestamp: datetime

type ChangeStats = dict[str, float]


@dataclass
class PerformanceRun:
    _id: str
    blobName: str
    timestamp: datetime
    changeStats: ChangeStats


@dataclass
class TuningRun(PerformanceRun):
    completed: bool = False
    hasArtifact: bool = False


@dataclass
class TuningConfig:
    _id: str
    timestamp: datetime
    run_id: str
    kernel_name: str
    result: dict[str, Any]


@dataclass
class BenchmarkRun(PerformanceRun):
    # _id: str
    headSha: str
    status: str
    conclusion: str
    numSteps: int
    steps: list[dict]
    hasArtifact: bool = False


@dataclass
class RunArtifact:
    kernels: list[dict]
    mapping: BenchmarkRun


@dataclass
class Kernel:
    id: str
    kernelType: str
    name: str
    tag: str
    dtype: str
    allowedBackends: list[str]
    problem: dict[str, Any]


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
    repoName: str
    branchName: str
    description: Optional[str] = None


@dataclass
class RepoMerge(RepoModification):
    prId: str
