from datetime import datetime, timezone
import logging
import traceback
from typing import Any, Dict, Optional
from github.WorkflowJob import WorkflowJob
from github.WorkflowRun import WorkflowRun

from backend.github_utils.auth import get_repo
from backend.runs import RunType, get_artifact_parser
from backend.runs.run_utils import RUN_INCOMPLETE_STATUSES
from backend.runs.workflows import find_workflow
from backend.storage.auth import get_blob_client
from backend.storage.types import WorkflowRunDb, WorkflowRunState

logger = logging.getLogger(__name__)


class RunTracker:
    def __init__(
        self,
        run_id: str,
        run_type: RunType,
        run_data: Optional[WorkflowRunState] = None,
    ):
        self._blob_client = get_blob_client()
        self._bench_repo = get_repo("bench")
        self._artifact_parser = get_artifact_parser(run_type)
        self._run_db = WorkflowRunDb

        workflow_info = find_workflow(run_type)
        if not workflow_info:
            raise ValueError(f"Run type {run_type} unsupported")

        self._identifier = workflow_info.identifier
        self._main_job = workflow_info.main_job
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
        logger.debug(f"Loading run_{self._run_id}")
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
                logger.warning(
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

        logger.debug(f"Updating run_{self._run_id} in db")
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
                    logger.error(
                        f"Failed to mark run_{self._run_id} with saved artifact",
                        "".join(traceback.format_exception(e)),
                    )
                    return False
                self._run.hasArtifact = True
            return save_success

        logger.error(f"No artifact returned by run_{self._run_id}")
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


def get_run_tracker(run: WorkflowRunState):
    try:
        run_type = RunType[run.type]
    except:
        raise ValueError(f"Workflow run type '{run.type}' not supported")

    return RunTracker(
        run_id=run._id,
        run_type=run_type,
        run_data=run,
    )
