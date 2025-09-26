from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
from .repository import create_repository


type ChangeStats = dict[str, float]


@dataclass
class WorkflowRunBase:
    _id: str
    blobName: str
    timestamp: datetime
    status: str
    conclusion: str
    numSteps: int
    steps: list[dict]
    completed: bool = False
    hasArtifact: bool = False
    mappingId: Optional[str] = None


@dataclass
class TuningRun(WorkflowRunBase):
    pass


@dataclass
class TuningConfig:
    _id: str
    timestamp: datetime
    run_id: str
    kernel_name: str
    result: dict[str, Any]


@dataclass
class BenchmarkRun(WorkflowRunBase):
    changeStats: ChangeStats = field(default_factory=dict)


@dataclass
class Kernel:
    id: str
    kernelType: str
    name: str
    tag: str
    dtype: str
    allowedBackends: list[str]
    problem: dict[str, Any]
    _id: str = None


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


TuningRunDb = create_repository(TuningRun, "tuningruns")
"""Repository for tuning run data with full type safety."""

TuningConfigDb = create_repository(TuningConfig, "tuningconfigs")
"""Repository for tuning configuration data with full type safety."""

BenchmarkRunDb = create_repository(BenchmarkRun, "runresults")
"""Repository for benchmark run results with full type safety."""

KernelDb = create_repository(Kernel, "kernels")
"""Repository for kernel data with full type safety."""

RepoPullRequestDb = create_repository(RepoPullRequest, "repomodifications")
"""Repository for repository pull request data with full type safety."""

RepoMergeDb = create_repository(RepoMerge, "repomodifications")
"""Repository for repository merge data with full type safety."""

PerformanceDb = create_repository(BenchmarkRun, "performanceruns")
