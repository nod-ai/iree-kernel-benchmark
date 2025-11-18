import argparse
import os
import shutil
import tempfile
import glob
from pathlib import Path
from backend.storage.auth import get_blob_client
from backend.storage.artifacts import *
from backend.storage.types import *


def main():
    parser = argparse.ArgumentParser(description="Upload benchmark results to Azure")
    parser.add_argument(
        "--dashboard-name", required=True, help="Name of the dashboard/Azure folder"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the directory containing JSON result files",
    )

    args = parser.parse_args()

    # Create temporary directory structure
    temp_base = tempfile.gettempdir()
    temp_path = os.path.join(
        temp_base, "test", args.dashboard_name, "benchmark-results", "gemm"
    )

    # Create the directory structure
    os.makedirs(temp_path, exist_ok=True)

    # Copy all JSON files from results-dir to the temp path
    json_files = glob.glob(os.path.join(args.results_dir, "*.json"))
    for json_file in json_files:
        shutil.copy2(json_file, temp_path)

    print(f"Copied {len(json_files)} JSON files to {temp_path}")

    # Get the path to benchmark-results folder
    benchmark_results_path = os.path.join(
        temp_base, "test", args.dashboard_name, "benchmark-results"
    )

    dir_client = get_blob_client()

    azure_folder = args.dashboard_name
    dir_client.rm(azure_folder, recursive=True)
    dir_client.upload(benchmark_results_path, azure_folder)

    print(f"Uploaded {benchmark_results_path} to Azure folder: {azure_folder}")

    print(f"View dashboard at http://51.8.231.84/dashboard/{azure_folder}")


if __name__ == "__main__":
    main()
