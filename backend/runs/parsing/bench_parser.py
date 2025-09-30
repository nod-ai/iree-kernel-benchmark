import json
import logging
import os
from pathlib import Path
import traceback
from typing import Dict, List, override
from uuid import uuid4

import pandas as pd

from backend.storage.artifacts import compare_artifact_kernels
from backend.storage.auth import get_blob_client
from backend.storage.types import BenchChangeStats, ChangeStatDb, WorkflowRunDb
from backend.storage.utils import convert_dict_case
from .artifact_parsing import RunArtifactParser


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
            self._logger.debug(f"Uploading artifacts to azure path {blob_name}")
            dir_client.upload(f"{local_path}/benchmark-results", blob_name)
        except:
            self._logger.error(f"Blob {blob_name} already exists. Skipped upload")
            return False

        baseline_kernels = self.load_data("baseline")
        if not baseline_kernels:
            self._logger.error(f"Failed to load baseline kernels for comparison")
            return False

        try:
            change_stats = compare_artifact_kernels(baseline_kernels, artifact_data)
            ChangeStatDb.upsert(
                BenchChangeStats(
                    _id=run_id,
                    runId=run_id,
                    changeStats=change_stats,
                )
            )
            return True
        except Exception as e:
            self._logger.error(
                f"Failed to calculate change statistics for run {run_id}",
                "".join(traceback.format_exception(e)),
            )
            return False


def parse_bench_kernels_from_path(artifact_path: Path) -> List[Dict]:
    results = []

    artifact_path = artifact_path / "benchmark-results"

    sub_dirs = os.listdir(artifact_path)
    if "json" in sub_dirs:
        sub_dirs = os.listdir(artifact_path / "json")
        artifact_path = artifact_path / "json"
    elif "csv" in sub_dirs:
        sub_dirs = os.listdir(artifact_path / "csv")
        artifact_path = artifact_path / "csv"

    logger = logging.getLogger("backend")

    logger.debug(f"Artifact path: {artifact_path}")

    for kernel_dir in sub_dirs:
        logger.debug(f"Kernel dir: {kernel_dir}")
        kernel_type = os.path.basename(kernel_dir)
        kernel_dir_path = artifact_path / kernel_type

        for result_file in os.listdir(kernel_dir_path):
            result_fname = os.path.basename(result_file)
            logger.debug(f"File: {result_fname}")
            result_file_path = kernel_dir_path / result_fname
            file_type = result_fname.split(".")[1]

            if file_type == "csv":
                backend_name = result_fname.split(".")[0].split(f"{kernel_type}_")[1]
                if backend_name == "wavegqa":
                    continue
                result_data = load_bench_result_csv_legacy(
                    backend_name, kernel_type, result_file_path
                )
            elif file_type == "json":
                result_data = load_bench_result_json(result_file_path)
            else:
                continue
            results.extend(result_data)

    if len(results) == 0:
        raise RuntimeError(
            f"Could not find kernels in local artifact directory {artifact_path}"
        )
    return results


def load_bench_result_json(json_path: str) -> List[Dict]:
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


def load_bench_result_csv_legacy(
    backend: str, kernel_type: str, csv_path: str
) -> List[Dict]:
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        shape = {}
        dtype = ""

        try:
            if kernel_type == "gemm":
                dtype = row["dtype"]
                shape = {
                    "M": row["M"],
                    "N": row["N"],
                    "K": row["K"],
                    "transpose": row["tA"] + row["tB"],
                }
            elif kernel_type == "attention":
                dtype = row["dtype"]
                shape = {
                    "B": row["B"],
                    "M": row["M"],
                    "N": row["N"],
                    "K1": row["K1"],
                    "K2": row["K2"],
                }
            elif kernel_type == "conv":
                dtype = row["input_dtype"]
                shape = {
                    "N": row["N"],
                    "H": row["H"],
                    "W": row["W"],
                    "C": row["C"],
                    "P": row["P"],
                    "Q": row["Q"],
                    "F": row["F"],
                    "S": row["S"],
                    "OP": row["OP"],
                }
        except:
            continue

        kernel = {
            "id": str(uuid4()),
            "machine": row.get("machine") or "MI325X",
            "backend": row.get("backend") or backend,
            "kernelType": row.get("kernel_type") or kernel_type,
            "dtype": dtype,
            "shape": shape,
            "name": row.get("name"),
            "tag": row.get("tag"),
            "meanMicroseconds": row.get("mean_microseconds"),
            "arithmeticIntensity": row.get("arithmetic_intensity"),
            "tflops": row.get("tflops"),
        }
        results.append(kernel)

    return results
