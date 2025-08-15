from datetime import timezone
import json

from numpy import extract
from globals import TESTING_MODE
from .directory import DirectoryClient
from .db import DatabaseClient
from .types import *
from .utils import convert_dict_case, get_nested_files
from auth import get_azure_clients, get_github_token, get_repo
from github import Repository, Artifact, WorkflowRun
from uuid import uuid4
from pathlib import Path
import os
import shutil
import requests
import zipfile
import pandas as pd
from typing import List, Dict


def load_result_csv(backend: str, kernel_type: str, csv_path: str) -> List[Dict]:
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


def load_result_json(backend: str, kernel_type: str, json_path: str) -> List[Dict]:
    with open(json_path, "r") as file:
        results = json.load(file)

    results = [
        {
            **convert_dict_case(result),
            "id": str(uuid4()),
            "backend": backend,
            "kernelType": kernel_type,
            "dtype": result["shape"].get("dtype") or result["shape"].get("input_dtype"),
        }
        for result in results
    ]

    return results


def load_tuning_result_json(json_path: os.PathLike, run_id: str) -> list[TuningConfig]:
    with open(json_path, "r") as file:
        results: dict[str, dict] = json.load(file)

    results = [
        TuningConfig(
            _id=str(uuid4()),
            timestamp=datetime.now(tz=timezone.utc),
            run_id=run_id,
            kernel_name=kernel_name,
            result=result,
        )
        for kernel_name, result in results.items()
    ]

    return results


def parse_kernels_from_path(artifact_path: str | Path) -> list[dict]:
    artifact_path = Path(artifact_path)

    results = []

    sub_dirs = os.listdir(artifact_path)
    if "json" in sub_dirs:
        sub_dirs = os.listdir(artifact_path / "json")
        artifact_path = artifact_path / "json"
    elif "csv" in sub_dirs:
        sub_dirs = os.listdir(artifact_path / "csv")
        artifact_path = artifact_path / "csv"

    for kernel_dir in sub_dirs:
        kernel_type = os.path.basename(kernel_dir)
        kernel_dir_path = artifact_path / kernel_type

        for result_file in os.listdir(kernel_dir_path):
            result_fname = os.path.basename(result_file)
            result_file_path = kernel_dir_path / result_fname
            backend_name = result_fname.split(".")[0].split(f"{kernel_type}_")[1]
            file_type = result_fname.split(".")[1]

            if backend_name == "wavegqa":
                continue

            if file_type == "csv":
                result_data = load_result_csv(
                    backend_name, kernel_type, result_file_path
                )
            elif file_type == "json":
                result_data = load_result_json(
                    backend_name, kernel_type, result_file_path
                )
            else:
                continue
            results.extend(result_data)

    return results


def parse_tuning_results_from_path(
    artifact_path: os.PathLike, run_id: str
) -> list[TuningConfig]:
    artifact_path = Path(artifact_path)

    results: list[TuningConfig] = []
    for result_json in get_nested_files(artifact_path, "json"):
        results.extend(load_tuning_result_json(result_json, run_id))

    return results


def download_artifact(
    artifact: Artifact.Artifact,
    local_path: os.PathLike = None,
    extract_name: str = "benchmark-results",
) -> Optional[Path]:
    local_path = Path(local_path) if local_path else Path(f"./tmp/{artifact.id}")

    download_url = artifact.archive_download_url
    headers = dict(artifact.raw_headers)
    headers["Authorization"] = f"Bearer {get_github_token('bench')}"

    response = requests.get(download_url, headers=headers)

    if response.status_code != 200:
        print("failed to download", artifact.id, download_url, response.json())
        None

    zip_path = local_path / "results.zip"
    extract_path = local_path / extract_name

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return local_path


def download_artifact_kernels(
    artifact: Artifact.Artifact, local_path: str = None
) -> tuple[List[Dict], Path]:

    kernels = []

    local_path = download_artifact(
        artifact, local_path, extract_name="benchmark-results"
    )
    if not local_path:
        return []
    extract_path = local_path / "benchmark-results"

    try:
        kernels = parse_kernels_from_path(extract_path)
    except Exception as e:
        print(f"Failed to load artifact {artifact.url}: {e}")

    return kernels, local_path


def download_artifact_tuning_results(
    artifact: Artifact.Artifact, local_path: os.PathLike = None
) -> tuple[List[TuningConfig], Path]:

    results = []

    base_path = download_artifact(artifact, local_path, extract_name="tuning-results")
    if not base_path:
        return []

    try:
        results = parse_tuning_results_from_path(
            base_path, run_id=str(artifact.workflow_run.id)
        )
    except Exception as e:
        print(f"Failed to load artifact {artifact.url}: {e}")

    return results, local_path


def save_results_from_local_path(
    dir_client: DirectoryClient,
    local_path: os.PathLike,
    blob_name: str,
    delete_local=True,
):
    try:
        print(f"Uploading artifacts to azure path {blob_name}")
        dir_client.upload(f"{local_path}/benchmark-results", blob_name)
        if delete_local:
            shutil.rmtree(local_path)
        return blob_name
    except:
        print(f"Blob {blob_name} already exists. Skipped upload")
        return None


def save_tuning_results_from_local_path(
    dir_client: DirectoryClient,
    local_path: os.PathLike,
    blob_name: str,
    delete_local=True,
):
    try:
        print(f"Uploading artifacts to azure path {blob_name}")
        dir_client.upload(f"{local_path}/tuning-results", blob_name)
        if delete_local:
            shutil.rmtree(local_path)
        return blob_name
    except:
        print(f"Blob {blob_name} already exists. Skipped upload")
        return None


def save_run_artifact(
    repo: Repository.Repository, run_data: BenchmarkRun, dir_client: DirectoryClient
) -> Optional[os.PathLike]:
    run = repo.get_workflow_run(int(run_data._id))
    artifacts = run.get_artifacts()

    for artifact in artifacts:
        artifact_kernels, artifact_path = download_artifact_kernels(artifact)
        print(f"Downloaded artifacts to local path {artifact_path}")

        if len(artifact_kernels) == 0:
            print("Failed to parse artifact")
            return None

        save_results_from_local_path(dir_client, artifact_path, run_data.blobName)

    print("No artifact returned by run")
    return None


def save_tuning_run_artifact(tuning_run_data: TuningRun) -> bool:
    repo = get_repo("bench")
    db_client, dir_client = get_azure_clients()

    run = repo.get_workflow_run(int(tuning_run_data._id))
    artifacts = run.get_artifacts()

    for artifact in artifacts:
        artifact_results, artifact_path = download_artifact_tuning_results(artifact)
        print(f"Downloaded artifacts to local path {artifact_path}")

        if len(artifact_results) == 0:
            print("Failed to parse artifact")
            return None

        db_client.insert_tuning_configs(artifact_results)
        print(f"Saved {len(artifact_results)} tuning results to database")

        save_tuning_results_from_local_path(
            dir_client, artifact_path, tuning_run_data.blobName
        )
        print(f"Saved {len(artifact_results)} tuning results to blob storage")

    print("No artifact returned by run")
    return None


def load_artifact_kernels(client: DirectoryClient, directory_name: str) -> list[dict]:
    artifact_id = str(uuid4())
    local_path = Path(f"./tmp/{artifact_id}")
    client.download(directory_name, str(local_path))

    artifact_path = local_path / "benchmark-results"
    results = parse_kernels_from_path(artifact_path)

    print(local_path)
    shutil.rmtree(local_path)
    return results


def compare_artifact_kernels(
    old: List[Dict], new: List[Dict] = None
) -> dict[str, float]:
    if not new:
        unique_kernel_types = {
            kernel["kernelType"] for kernel in old if kernel["backend"] == "wave"
        }
        return {k_type: 0.0 for k_type in unique_kernel_types}

    def hash_kernel(kernel: Dict) -> str:
        shape = kernel["shape"]
        dtype = kernel["dtype"]
        shape_hash = "_".join([f"{k}.{v}" for k, v in shape.items()])
        return f"{shape_hash}_dt{dtype}"

    old_wave = [k for k in old if k["backend"] == "wave"]
    new_wave = [k for k in new if k["backend"] == "wave"]

    old_shapes = {hash_kernel(k) for k in old_wave}
    common_shapes = {hash_kernel(k) for k in new_wave if hash_kernel(k) in old_shapes}

    kernel_types = {k["kernelType"] for k in new_wave}
    perf_stats = {k_type: [0, 0, 0] for k_type in kernel_types}

    for i, kernel_list in enumerate((old_wave, new_wave)):
        for kernel in kernel_list:
            kernel_hash = hash_kernel(kernel)
            if kernel_hash not in common_shapes:
                continue
            k_type = kernel["kernelType"]
            if k_type not in perf_stats:
                continue
            runtime = kernel["meanMicroseconds"]
            perf_stats[k_type][i] += runtime
            if i == 0:
                perf_stats[k_type][2] += 1

    change_stats = {}
    for k_type, (old_perf, new_perf, count) in perf_stats.items():
        if old_perf > 0:
            change_stats[k_type] = (new_perf / old_perf - 1) * 100
        else:
            change_stats[k_type] = 0.0

    return change_stats


def fill_new_kernels(old: List[Dict], new: List[Dict]) -> List[Dict]:
    new_wave = [k for k in new if "wave" in k["backend"]]
    old_other = [k for k in old if "wave" not in k["backend"]]

    return new_wave + old_other
