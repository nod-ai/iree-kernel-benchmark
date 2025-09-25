from .types import *
from backend.github_utils import get_github_token
from github import Artifact
from uuid import uuid4
from pathlib import Path
import os
import requests
import zipfile
from typing import List, Dict


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


def compare_artifact_kernels(
    old: List[Dict], new: List[Dict] = None, backend: str = "wave"
) -> dict[str, float]:
    if not new:
        unique_kernel_types = {
            kernel["kernelType"] for kernel in old if kernel["backend"] == backend
        }
        return {k_type: 0.0 for k_type in unique_kernel_types}

    def hash_kernel(kernel: Dict) -> str:
        shape = kernel["shape"]
        dtype = kernel["dtype"]
        shape_hash = "_".join([f"{k}.{v}" for k, v in shape.items()])
        return f"{shape_hash}_dt{dtype}"

    old_wave = [k for k in old if k["backend"] == backend]
    new_wave = [k for k in new if k["backend"] == backend]

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
