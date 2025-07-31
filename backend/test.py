from auth import get_repo
from github import Github, Auth
from storage.artifacts import *

repo = get_repo("test")

# kernels = download_artifact_kernels_by_run_id(repo, '16468922095')
kernels = download_all_artifact_kernels(repo, limit=1)
print(f"{len(kernels)}x{len(kernels[0])} kernels loaded successfully")
