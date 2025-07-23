from .directory import DirectoryClient
from .db import DatabaseClient
from .types import *
from auth import get_access_token
from github import Repository
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

def download_artifact_kernels(repo: Repository.Repository, limit=10) -> List[List[Dict]]:
    results = []

    artifacts = repo.get_artifacts()
    for artifact in artifacts:
        download_url = artifact.archive_download_url
        headers = dict(artifact.raw_headers)
        headers['Authorization'] = f'Bearer {get_access_token()}'

        response = requests.get(download_url, headers=headers)
        
        if response.status_code != 200:
            continue

        artifact_id = uuid4()
        base_path = Path(f"./tmp/{artifact_id}")
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
            results.append(kernels)
        except:
            print(f'Failed to load artifact {artifact.url}')
        
        shutil.rmtree(base_path)
        if len(results) >= limit:
            return results
    
    return results

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