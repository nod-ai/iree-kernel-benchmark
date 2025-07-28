from globals import TESTING_MODE
from .directory import DirectoryClient
from .db import DatabaseClient
from .types import *
from auth import get_access_token
from github import Repository, Artifact
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
                "B": row["B"],
                "H": row["H"],
                "W": row["W"],
                "C": row["C"],
                "P": row["P"],
                "Q": row["Q"],
                "F": row["F"],
                "S": row["S"],
            }

        kernel = {
            "id": str(uuid4()),
            "backend": backend,
            "kernelType": kernel_type,
            "dtype": dtype,
            "shape": shape,
            "name": row["name"],
            "tag": row["tag"],
            "meanMicroseconds": row["mean_microseconds"],
            "arithmeticIntensity": row["arithmetic_intensity"],
            "tflops": row["tflops"],
        }
        results.append(kernel)

    return results

def parse_kernels_from_path(artifact_path: str | Path) -> list[dict]:
    artifact_path = Path(artifact_path)

    results = []

    for kernel_dir in os.listdir(artifact_path):
        kernel_type = os.path.basename(kernel_dir)
        kernel_dir_path = artifact_path / kernel_type

        for csv_file in os.listdir(kernel_dir_path):
            csv_name = os.path.basename(csv_file)
            csv_file_path = kernel_dir_path / csv_name
            backend_name = csv_name.split('.')[0].split(f'{kernel_type}_')[1]
            
            if backend_name == 'wavegqa':
                continue

            result_data = load_result_csv(backend_name, kernel_type, csv_file_path)
            results.extend(result_data)

    return results

def download_artifact_kernels(
        artifact: Artifact.Artifact, 
        local_path: str = None) -> tuple[List[Dict], Path]:
    
    kernels = []

    local_path = local_path or f"./tmp/{artifact.id}"

    download_url = artifact.archive_download_url
    headers = dict(artifact.raw_headers)
    headers['Authorization'] = f'Bearer {get_access_token('TEST' if TESTING_MODE else 'BENCH')}'

    response = requests.get(download_url, headers=headers)
    
    if response.status_code != 200:
        print('failed to download', artifact.id, download_url, response.json())
        return []

    base_path = Path(local_path)
    zip_path = base_path / "results.zip"
    extract_path = base_path / "benchmark-results"

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    try:
        kernels = parse_kernels_from_path(extract_path)
    except:
        print(f'Failed to load artifact {artifact.url}')
    
    return kernels, base_path

def save_all_artifact_kernels(repo: Repository.Repository, dir_client: DirectoryClient, limit=10) -> tuple[List[List[Dict]], List[str]]:
    results = []
    blob_paths = []

    artifacts = repo.get_artifacts()
    for artifact in artifacts:
        artifact_kernels, artifact_path = download_artifact_kernels(artifact, f'./tmp/{artifact.id}')
        results.append(artifact_kernels)

        blobName = str(artifact.id)
        blob_paths.append(blobName)
        try:
            dir_client.upload(f'{artifact_path}/benchmark-results', blobName)
        except:
            print(f'Blob already exists for artifact {artifact.id}. Continuing...')
        
        shutil.rmtree(artifact_path)

        if len(results) >= limit:
            return results, blob_paths
    
    return results, blob_paths

def download_all_artifact_kernels(repo: Repository.Repository, limit=10) -> List[List[Dict]]:
    results = []

    artifacts = repo.get_artifacts()
    for artifact in artifacts:
        artifact_kernels, artifact_path = download_artifact_kernels(artifact)
        results.append(artifact_kernels)
        shutil.rmtree(artifact_path)

        if len(results) >= limit:
            return results
    
    return results

def download_artifact_kernels_by_run_id(repo: Repository.Repository, run_id) -> List[Dict]:
    run = repo.get_workflow_run(int(run_id))
    artifacts = run.get_artifacts()
    for artifact in artifacts:
        artifact_kernels, artifact_path = download_artifact_kernels(artifact)
        
        return artifact_kernels
    return []

def save_run_artifact(repo: Repository.Repository, run_data: BenchmarkRun, dir_client: DirectoryClient) -> Optional[os.PathLike]:
    run = repo.get_workflow_run(int(run_data._id))
    artifacts = run.get_artifacts()
    
    for artifact in artifacts:
        artifact_kernels, artifact_path = download_artifact_kernels(artifact)
        print(f'Downloaded artifacts to local path {artifact_path}')

        if len(artifact_kernels) == 0:
            print('Failed to parse artifact')
            return None

        try:
            print(f'Uploading artifacts to azure path {run_data.blobName}')
            dir_client.upload(f'{artifact_path}/benchmark-results', run_data.blobName)
            shutil.rmtree(artifact_path)
            return run_data.blobName
        except:
            print(f'Blob already exists for artifact {artifact.id}. Skipped upload')
            return None
    
    print('No artifact returned by run')
    return None

def load_artifact_kernels(client: DirectoryClient, directory_name: str) -> list[dict]:
    artifact_id = str(uuid4())
    local_path = Path(f'./tmp/{artifact_id}')
    client.download(directory_name, str(local_path))

    artifact_path = local_path / 'benchmark-results'
    results = parse_kernels_from_path(artifact_path)

    shutil.rmtree(local_path)
    return results

def fetch_all_artifacts(directory_client: DirectoryClient, db_client: DatabaseClient) -> list[RunArtifact]:
    artifacts = []

    runs = db_client.find_all_runs()
    for run in runs:
        run_dir_name = run.blobName
        kernels = load_artifact_kernels(directory_client, f'{run_dir_name}/benchmark-results')
        artifacts.append(RunArtifact(kernels, run))
    
    return artifacts

def fetch_latest_artifact(directory_client: DirectoryClient, db_client: DatabaseClient) -> Optional[RunArtifact]:
    artifacts = fetch_all_artifacts(directory_client, db_client)
    return artifacts[0] if len(artifacts) > 0 else None

def fetch_artifact_by_trigger_id(directory_client: DirectoryClient, db_client: DatabaseClient, trigger_id: str) -> Optional[RunArtifact]:
    runs = db_client.query_runs(f'triggerId eq {trigger_id}')
    if len(runs) == 0:
        return None
    
    run_dir_name = runs[0].blobName
    kernels = load_artifact_kernels(directory_client, f'{run_dir_name}/benchmark-results')
    return RunArtifact(kernels, runs[0])

def compare_artifact_kernels(old: List[Dict], new: List[Dict] = None) -> dict[str, float]:
    if not new:
        unique_backends = set([kernel['backend'] for kernel in new])
        return { backend: 0.0 for backend in unique_backends }

    def hash_kernel(kernel: Dict) -> str:
        shape: dict = kernel['shape']
        dtype: str = kernel['dtype']
        shape_hash = '_'.join([f'{k}.{v}' for k, v in shape.items()])
        return f'{shape_hash}_dt{dtype}'

    common_shapes = { hash_kernel(kernel) for kernel in old + new }

    unique_backends = set([kernel['backend'] for kernel in new])
    perf_stats = { backend: [0, 0, 0] for backend in unique_backends }

    for i, kernel_list in enumerate((old, new)):
        for kernel in kernel_list:
            kernel_hash = hash_kernel(kernel)
            if kernel_hash not in common_shapes:
                continue
            backend = kernel['backend']
            if backend not in perf_stats:
                continue
            runtime = kernel['meanMicroseconds']
            perf_stats[backend][i] += runtime
            if i == 0:
                perf_stats[backend][2] += 1
    
    change_stats = { backend: 0.0 for backend in unique_backends }
    for backend, (old_perf, new_perf, count) in perf_stats.items():
        change_stats[backend] = (new_perf / old_perf - 1) * 100
    
    return change_stats