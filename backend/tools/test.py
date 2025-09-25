from backend.storage.auth import get_azure_clients
from backend.github_utils import get_repo
from backend.storage.artifacts import *
from backend.storage.types import *

from backend.storage.rebase import rebase_performance
from github import Github, Auth
import json
from dotenv import load_dotenv
from pathlib import Path
from dataclass_wizard import asdict
import os

db_client, dir_client = get_azure_clients()

# db_client.clear_all_kernels()
# with open("test/configs.json", "r") as file:
#     kernels = json.load(file)

# db_client.insert_kernels([Kernel(**k) for k in kernels])

azure_folder = "tuning"
dir_client.rm(azure_folder, recursive=True)
dir_client.upload(f"test/benchmark-results", azure_folder)

# kernels = parse_kernels_from_path(
#     "./tmp/5a329c8b-ee87-4d24-97de-c994e0b858c4/benchmark-results"
# )
# print(any(k["backend"] == "wavenew" for k in kernels))

# db_client.delete_runs("conclusion eq 'failure' or conclusion eq 'cancelled'")

# old = parse_kernels_from_path("./test/benchmark-results-mi300")
# new = parse_kernels_from_path("./test/benchmark-results-mi325")

# print(compare_artifact_kernels(old, new))

# rebase_performance(20)
