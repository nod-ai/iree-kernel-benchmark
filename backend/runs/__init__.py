import os
from enum import Enum
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
