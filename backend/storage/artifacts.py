from .directory import DirectoryClient
from .db import DatabaseClient
from uuid import uuid4
from pathlib import Path
import os
import shutil
import pandas as pd
import uuid
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
            "id": str(uuid.uuid4()),
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


def load_artifact(client: DirectoryClient, directory_name: str) -> list[dict]:
    artifact_id = str(uuid4())
    local_path = Path(f'./tmp/{artifact_id}')
    client.download(directory_name, str(local_path))

    artifact_path = local_path / 'benchmark-results'

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

    shutil.rmtree(local_path)

    return results

def fetch_latest_artifact(directory_client: DirectoryClient, db_client: DatabaseClient) -> list[dict]:
    runs = db_client.find_all_runs()
    if len(runs) == 0:
        return []
    
    latest_run = runs[0]
    run_dir_name = latest_run.get('blobName')
    kernels = load_artifact(directory_client, f'{run_dir_name}/benchmark-results')
    return kernels
