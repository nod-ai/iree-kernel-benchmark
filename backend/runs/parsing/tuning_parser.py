from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import traceback
from typing import Dict, List, Optional, override
from uuid import uuid4

from dataclass_wizard import asdict, fromdict

from backend.github_utils.gist import update_gist
from backend.storage.auth import get_blob_client
from backend.storage.directory import DirectoryClient
from backend.storage.types import TuningConfig, TuningConfigDb
from backend.storage.utils import get_nested_files
from .artifact_parsing import RunArtifactParser

logger = logging.getLogger(__name__)


class TuningArtifactParser(RunArtifactParser):
    @override
    def _parse_from_local_path(self, local_path):
        return parse_tuning_results_from_path(local_path)

    @override
    def _save_artifact(self, local_path, artifact_data: list[TuningConfig], run):
        if len(artifact_data) == 0:
            logger.error("Failed to parse artifact")
            return False

        for i in range(len(artifact_data)):
            artifact_data[i]._id = f"{run._id}-{i}"
            artifact_data[i].run_id = run._id
        upsert_success = TuningConfigDb.upsert_many(artifact_data)
        if upsert_success:
            logger.debug(f"Saved {len(artifact_data)} tuning results to database")
        else:
            logger.error(f"Failed to save tuned configs")
            return False

        updated_configs = TuningConfigDb.find_all()
        updated_configs_json = [asdict(config) for config in updated_configs]
        updated_gist = update_gist(os.getenv("TUNING_GIST_ID"), updated_configs_json)
        if updated_gist:
            logger.debug(
                f"Saved updated tuning results to gist {updated_gist.gist_url}"
            )

        return True


def parse_tuning_results_from_path(
    artifact_path: os.PathLike, run_id: Optional[str] = None
) -> list[TuningConfig]:
    artifact_path = Path(artifact_path)

    results: list[TuningConfig] = []
    for result_json in get_nested_files(artifact_path, "json"):
        results.extend(load_tuning_result_json(result_json, run_id))

    return results


def load_tuning_result_json(
    json_path: os.PathLike, run_id: Optional[str] = None
) -> List[TuningConfig]:
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
