import os
from enum import Enum

from backend.storage.repository import DatabaseRepository
from backend.storage.types import BenchmarkRunDb, TuningRunDb, WorkflowRunBase
from .parsing.bench_parser import BenchmarkArtifactParser
from .parsing.tuning_parser import TuningArtifactParser


class RunType(Enum):
    BENCHMARK = 0
    TUNING = 1


def get_artifact_parser(run_type: RunType, tmp_dir: os.PathLike = "./tmp"):
    match run_type:
        case RunType.BENCHMARK:
            return BenchmarkArtifactParser(tmp_dir)
        case RunType.TUNING:
            return TuningArtifactParser(tmp_dir)
        case _:
            raise ValueError(f"Received invalid run type {run_type}")


def get_run_db(run_type: RunType) -> DatabaseRepository[WorkflowRunBase]:
    match run_type:
        case RunType.BENCHMARK:
            return BenchmarkRunDb
        case RunType.TUNING:
            return TuningRunDb
        case _:
            raise ValueError(f"Received invalid run type {run_type}")
