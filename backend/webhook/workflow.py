from auth import get_access_token
from storage.db import DatabaseClient
from storage.directory import DirectoryClient
from storage.types import *
from storage.artifacts import download_artifact_kernels_by_run_id
from dataclass_wizard import asdict
from github import Github, Auth
import json

def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)

class WorkflowListener:
    def __init__(self, db_client: DatabaseClient, storage_client: DirectoryClient):
        auth = Auth.Token(get_access_token('BENCH'))
        gh = Github(auth=auth)
        self._repo = gh.get_repo('nod-ai/iree-kernel-benchmark')
        self._db_client = db_client
        self._storage_client = storage_client

    def _handle_workflow_run_requested(self, run_payload: dict):
        run_data = run_payload['workflow_run']
        run_id = str(run_data['id'])
        run = BenchmarkRun(
            _id=run_id,
            headSha=run_data['head_sha'],
            status=run_data['status'],
            conclusion=run_data['conclusion'] or 'unknown',
            steps=[],
            blobName=run_data['updated_at'],
            kernels=[],
        )
        print('adding new run', run_id, jsonify(run))
        self._db_client.insert_run(run)

    def _handle_workflow_run_progress(self, run_payload: dict):
        run_data = run_payload['workflow_run']
        run_id = str(run_data['id'])

        print('updating run', run_id, json.dumps({
            'status': run_data['status'],
            'conclusion': run_data['conclusion'] or 'unknown',
            'updated_at': run_data['updated_at'],
        }, indent=4))

        self._db_client.update_run(
            run_id,
            status=run_data['status'],
            conclusion=run_data['conclusion'],
            blobName=run_data['updated_at'],
        )

    def _handle_workflow_run_complete(self, run_payload: dict):
        run_data = run_payload['workflow_run']
        run_id = str(run_data['id'])

        print('completing run', run_id, json.dumps({
            'status': run_data['status'],
            'conclusion': run_data['conclusion'] or 'unknown',
            'updated_at': run_data['updated_at'],
        }, indent=4))

        kernels = download_artifact_kernels_by_run_id(self._repo, run_id)

        self._db_client.update_run(
            run_id,
            status=run_data['status'],
            conclusion=run_data['conclusion'],
            blobName=run_data['updated_at'],
            kernels=kernels,
        )

    def handle_workflow_run_payload(self, run_payload: dict):
        if run_payload['repository']['name'] != 'github-api-test':
            return
        if run_payload['workflow_run']['name'] != 'Test Job':
            return
        
        handler = {
            'requested': self._handle_workflow_run_requested,
            'in_progress': self._handle_workflow_run_progress,
            'completed': self._handle_workflow_run_complete,
        }

        handler[run_payload['action']](run_payload)

    def handle_workflow_job_payload(self, job_payload: dict):
        if job_payload['repository']['name'] != 'github-api-test':
            return
        if job_payload['workflow_job']['name'] != 'dummy-delay-job':
            return

        run_id = str(job_payload['workflow_job']['run_id'])
        steps = job_payload['workflow_job']['steps']

        print('updating job', json.dumps(steps, indent=4))

        self._db_client.update_run(run_id, steps=steps)
