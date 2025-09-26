from datetime import datetime, timezone
import json
import logging
import traceback
from typing import Any, Dict, Optional
from github.WorkflowJob import WorkflowJob
from github.WorkflowRun import WorkflowRun
from dataclass_wizard import fromdict

from backend.github_utils.auth import get_repo
from backend.runs import RunType, get_artifact_parser, get_run_db
from backend.runs.run_utils import RUN_INCOMPLETE_STATUSES, parse_run_json
from backend.storage.auth import get_blob_client
from backend.storage.types import (
    BenchmarkRun,
    BenchmarkRunDb,
    TuningRun,
    WorkflowRunBase,
)


class RunTracker:
    def __init__(
        self,
        run_id: str,
        run_type: RunType,
        run_data: Optional[WorkflowRunBase] = None,
        main_job: Optional[str] = None,
        identifier: Optional[str] = None,
    ):
        self._blob_client = get_blob_client()
        self._bench_repo = get_repo("bench")
        self._logger = logging.getLogger("backend")
        self._artifact_parser = get_artifact_parser(run_type)
        self._run_db = get_run_db(run_type)

        self._identifier = identifier
        self._main_job = main_job
        self._run_id = run_id
        self._run_type = run_type

        if run_data:
            self._run = run_data
        else:
            try:
                self._load_data_from_db()
            except:
                raise ValueError(f"Could not find run {run_id} in database")

    def update(self):
        self._logger.debug(f"Loading run_{self._run_id}")
        gh_run = self._get_gh_run()
        gh_main_job = self._get_gh_job(gh_run)

        db_run_update = {
            "status": gh_run.status,
            "conclusion": gh_run.conclusion,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._identifier:
            try:
                gh_id_job = self._get_gh_job(gh_run, job_name="Identifier")
                if gh_id_job:
                    for job_step in gh_id_job.steps:
                        if job_step.name.startswith(f"{self._identifier}_"):
                            mapping_id = job_step.name.split(f"{self._identifier}_")[1]
                            db_run_update.update({"mappingId": mapping_id})
                            break
            except:
                self._logger.warning(
                    f"Failed to load identifier {self._identifier} for run_{self._run_id}"
                )
                pass

        if gh_main_job:
            job_steps = [
                {
                    "name": step.name,
                    "status": step.status,
                    "conclusion": step.conclusion,
                    "number": step.number,
                    "started_at": (
                        datetime.isoformat(step.started_at) if step.started_at else None
                    ),
                    "completed_at": (
                        datetime.isoformat(step.completed_at)
                        if step.completed_at
                        else None
                    ),
                }
                for step in gh_main_job.steps
            ]
            db_run_update.update(
                {
                    "numSteps": len(job_steps),
                    "steps": job_steps,
                }
            )

        self._logger.debug(f"Updating run_{self._run_id} in db")
        self._run_db.update_by_id(self._run_id, db_run_update)
        self._load_data_from_db()

    def run_id(self) -> str:
        return self._run_id

    def get_data(self) -> WorkflowRun:
        self._load_data_from_db()
        return self._run

    def is_complete(self) -> bool:
        self._load_data_from_db()
        return self._run.status not in RUN_INCOMPLETE_STATUSES

    def has_artifact(self) -> bool:
        self._load_data_from_db()
        return self._run.hasArtifact

    def get_artifact_data(self) -> Optional[Any]:
        if not self.is_complete():
            return None

        if not self._run.hasArtifact:
            save_success = self.save_artifact()
            if not save_success:
                return None

        return self._artifact_parser.load_data(self._run.blobName)

    def save_artifact(self) -> bool:
        gh_run = self._get_gh_run()
        artifacts = gh_run.get_artifacts()

        for artifact in artifacts:
            save_success, _ = self._artifact_parser.parse_and_save_artifact(
                artifact, self._run
            )
            if save_success:
                try:
                    self._run_db.update_by_id(self._run_id, {"hasArtifact": True})
                except Exception as e:
                    self._logger.error(
                        f"Failed to mark run_{self._run_id} with saved artifact",
                        "".join(traceback.format_exception(e)),
                    )
                    return False
                self._run.hasArtifact = True
            return save_success

        self._logger.error(f"No artifact returned by run_{self._run_id}")
        return False

    def _load_data_from_db(self):
        self._run = self._run_db.find_by_id(self._run_id)

    def _get_gh_run(self) -> Optional[WorkflowRun]:
        try:
            return self._bench_repo.get_workflow_run(int(self._run_id))
        except:
            return None

    def _get_gh_job(
        self,
        gh_run: Optional[WorkflowRun] = None,
        job_name: Optional[str] = None,
    ) -> Optional[WorkflowJob]:
        if not gh_run:
            gh_run = self._get_gh_run()
        if not job_name:
            job_name = self._main_job

        gh_jobs = gh_run.jobs()
        for gh_job in gh_jobs:
            if job_name.strip().upper() not in gh_job.name.strip().upper():
                continue
            return gh_job

        return None


def get_run_tracker(run: WorkflowRunBase):
    if isinstance(run, BenchmarkRun):
        return RunTracker(
            run_id=run._id,
            run_type=RunType.BENCHMARK,
            run_data=run,
            main_job="Short Benchmark",
            identifier="headSha",
        )
    elif isinstance(run, TuningRun):
        return RunTracker(
            run_id=run._id,
            run_type=RunType.TUNING,
            run_data=run,
        )
    else:
        raise ValueError("Workflow run type not supported")
