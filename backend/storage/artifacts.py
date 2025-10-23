from logging import getLogger
from .types import *
from backend.github_utils import get_github_token
from github import Artifact
from uuid import uuid4
from pathlib import Path
import os
import requests
import zipfile
from typing import List, Dict

logger = getLogger(__name__)


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
        logger.error(
            f"Failed to download artifact {artifact.id} at {download_url}:\n{response.json()}"
        )
        return None

    zip_path = local_path / "results.zip"
    extract_path = local_path / extract_name

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    with open(zip_path, "wb") as f:
        f.write(response.content)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    except:
        logger.error(f"Failed to parse artifact data for artifact {artifact.id}")
        return None

    return local_path


def fill_new_kernels(old: List[Dict], new: List[Dict]) -> List[Dict]:
    new_wave = [k for k in new if "wave" in k["backend"]]
    old_other = [k for k in old if "wave" not in k["backend"]]

    return new_wave + old_other
