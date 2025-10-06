import json
import logging
import os
from pathlib import Path
import traceback
from typing import Dict, List, override
from uuid import uuid4

import pandas as pd

from backend.perf.comparisons import compare_kernels
from backend.storage.auth import get_blob_client
from backend.storage.types import BenchChangeStats, ChangeStatDb, WorkflowRunDb
from backend.storage.utils import convert_dict_case, get_nested_files
from .artifact_parsing import RunArtifactParser

logger = logging.getLogger(__name__)


class BenchmarkArtifactParser(RunArtifactParser):
    @override
    def _parse_from_local_path(self, local_path):
        return parse_bench_kernels_from_path(local_path)

    @override
    def _save_artifact(self, local_path, artifact_data, run):
        dir_client = get_blob_client()
        blob_name = run.blobName
        run_id = run._id

        try:
            logger.debug(f"Uploading artifacts to azure path {blob_name}")
            dir_client.upload(f"{local_path}/benchmark-results", blob_name)
        except:
            logger.error(f"Blob {blob_name} already exists. Skipped upload")
            return False

        baseline_kernels = self.load_data("baseline")
        if not baseline_kernels:
            logger.error(f"Failed to load baseline kernels for comparison")
            return False

        try:
            old_perf, new_perf = compare_kernels(baseline_kernels, artifact_data)
            ChangeStatDb.upsert(
                BenchChangeStats(
                    _id=str(uuid4()),
                    runId=run_id,
                    machine="mi325x",
                    old=old_perf,
                    new=new_perf,
                )
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to calculate change statistics for run {run_id}",
                "".join(traceback.format_exception(e)),
            )
            return False


def parse_bench_kernels_from_path(artifact_path: Path) -> List[Dict]:
    results = []

    artifact_path = artifact_path / "benchmark-results"
    logger.debug(f"Artifact path: {artifact_path}")

    for result_json in get_nested_files(artifact_path, "json"):
        result_data = load_bench_result_json(result_json)
        results.extend(result_data)

    if len(results) == 0:
        raise RuntimeError(
            f"Could not find kernels in local artifact directory {artifact_path}"
        )
    return results


def load_bench_result_json(json_path: os.PathLike) -> List[Dict]:
    with open(json_path, "r") as file:
        results = json.load(file)

    results = [
        {
            **convert_dict_case(result),
            "id": str(uuid4()),
            "dtype": result["shape"].get("dtype") or result["shape"].get("input_dtype"),
        }
        for result in results
    ]

    return results
