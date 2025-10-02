from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
from .repository import create_repository


@dataclass
class WorkflowRunState:
    _id: str
    type: str
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
class TuningConfig:
    _id: str
    timestamp: datetime
    run_id: str
    kernel_name: str
    result: dict[str, Any]


@dataclass
class BenchChangeStats:
    _id: str
    runId: str
    old: dict[str, Any]
    new: dict[str, Any]


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


# @dataclass
# class RepoCommit:
#     _id: str
#     title: str
#     author: ChangeAuthor
#     timestamp: datetime
#     description: Optional[str] = None


@dataclass
class RepoPullRequest:
    _id: str
    url: str
    type: str
    timestamp: datetime
    author: ChangeAuthor
    title: str
    status: str
    commits: int
    repoName: str
    branchName: str
    mappingId: Optional[str] = None
    description: Optional[str] = None
    isMerged: bool = False


WorkflowRunDb = create_repository(WorkflowRunState, "workflowruns")
"""Repository for workflow run data with full type safety."""

ChangeStatDb = create_repository(BenchChangeStats, "benchchangestats")

TuningConfigDb = create_repository(TuningConfig, "tuningconfigs")
"""Repository for tuning configuration data with full type safety."""

KernelDb = create_repository(Kernel, "kernels")
"""Repository for kernel data with full type safety."""

RepoPullRequestDb = create_repository(RepoPullRequest, "repopullrequests")
"""Repository for repository pull request data with full type safety."""
