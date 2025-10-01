from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import shutil
import traceback
from typing import Any, Optional, Tuple
from uuid import uuid4
from github.Artifact import Artifact
import requests

from backend.github_utils.auth import get_github_token
from backend.storage.artifacts import download_artifact
from backend.storage.auth import get_blob_client
from backend.storage.types import WorkflowRunState

logger = logging.getLogger(__name__)


class RunArtifactParser(ABC):
    def __init__(self, local_tmp_dir: os.PathLike):
        os.makedirs(local_tmp_dir, exist_ok=True)
        self._local_tmp_dir = Path(local_tmp_dir)

    def load_data(self, blob_name: str) -> Optional[Any]:
        dir_client = get_blob_client()
        local_path = self._local_tmp_dir / str(uuid4())
        dir_client.download(blob_name, str(local_path))
        local_path = local_path / blob_name

        try:
            parsed_data = self._parse_from_local_path(local_path)
            shutil.rmtree(local_path)
            return parsed_data
        except Exception as e:
            logger.error(
                f"Failed to load artifact data: \n"
                "".join(traceback.format_exception(e))
            )
            shutil.rmtree(local_path)
            return None

    def parse_and_save_artifact(
        self, gh_artifact: Artifact, run: WorkflowRunState
    ) -> Tuple[bool, Any]:
        artifact_data, local_path = self.parse_artifact(gh_artifact)
        if not artifact_data:
            shutil.rmtree(local_path)
            return False, None

        try:
            save_success = self._save_artifact(local_path, artifact_data, run)
            if not save_success:
                raise RuntimeError("Failed to save artifact")

            logger.debug(f"Successfully saved data for artifact_{gh_artifact.id}")
            shutil.rmtree(local_path)
            return True, artifact_data

        except Exception as e:
            logger.error(
                f"Failed to save parsed artifact data: \n"
                "".join(traceback.format_exception(e))
            )
            shutil.rmtree(local_path)
            return False, artifact_data

    def parse_artifact(self, gh_artifact: Artifact) -> Tuple[Any, Path]:
        local_path = download_artifact(
            gh_artifact, self._local_tmp_dir / str(gh_artifact.id)
        )
        parsed_data = None

        if local_path:
            try:
                parsed_data = self._parse_from_local_path()
            except Exception as e:
                logger.error(
                    f"Failed to parse artifact data: \n"
                    "".join(traceback.format_exception(e))
                )

        return parsed_data, local_path

    @abstractmethod
    def _parse_from_local_path(self, local_path: Path) -> Any:
        pass

    @abstractmethod
    def _save_artifact(
        self, local_path: Path, artifact_data: Any, run: WorkflowRunState
    ) -> bool:
        pass
