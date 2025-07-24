from auth import get_access_token
from storage.db import DatabaseClient
from storage.directory import DirectoryClient
from storage.types import *
from storage.artifacts import download_artifact_kernels_by_run_id
from dataclass_wizard import asdict
from github import Github, Auth
import json
import requests

def jsonify(model) -> str:
    return json.dumps(asdict(model), indent=4)

class WaveUpdateListener:
    '''
    Wave Update Events:
    - assigned
    - auto_merge_disabled
    - auto_merge_enabled
    - closed
    - converted_to_draft
    - demilestoned
    - dequeued
    - edited
    - enqueued
    - labeled
    - locked
    - milestoned
    - opened
    - ready_for_review
    - reopened
    - review_request_removed
    - review_requested
    - synchronize
    - unassigned
    - unlabeled
    - unlocked
    '''

    def __init__(self, db_client: DatabaseClient, storage_client: DirectoryClient):
        self._wave_auth = Auth.Token(get_access_token('WAVE'))
        self._bench_auth = Auth.Token(get_access_token('BENCH'))
        gh_wave = Github(auth=self._wave_auth)
        gh_bench = Github(auth=self._bench_auth)
        self._wave_repo = gh_wave.get_repo('iree-org/wave')
        self._bench_repo = gh_bench.get_repo('nod-ai/iree-kernel-benchmark')
        self._db_client = db_client
        self._storage_client = storage_client

    def trigger_workflow(self, repo_name: str, branch_name: str, metadata: dict = None):
        workflow_id = 'short_bench.yml'
        token = self._bench_auth.token

        url = f'https://api.github.com/repos/nod-ai/iree-kernel-benchmark/actions/workflows/{workflow_id}/dispatches'
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        body = {
            "ref": "main",
            "inputs": {
                "iterations": "3",
                "max_kernels": "50",
                "selected_backend": "all",
                "pr_repository": repo_name,
                "pr_branch": branch_name,
            }
        }
        if metadata:
            body['inputs']['metadata'] = json.dumps(metadata)
        
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 204:
            print(f'Workflow failed: {response.json()}')
        print(f"Workflow dispatched successfully: {response.status_code}")
    
    def _save_pr(self, id: str, pr_obj: dict) -> dict:
        author = ChangeAuthor(
            name=pr_obj['user'].get('name') or pr_obj['user'].get('login', 'Anonymous'),
            profileUrl=pr_obj['user']['avatar_url']
        )
        if pr_obj['merged']:
            merge = RepoMerge(
                _id=id,
                headSha=pr_obj['head']['sha'],
                url=pr_obj['html_url'],
                type='merge',
                timestamp=datetime.fromisoformat(pr_obj['merged_at']),
                author=author,
                prId=str(pr_obj['id'])
            )
            self._db_client.insert_merge(merge)
            return asdict(merge)
        else:
            pr = RepoPullRequest(
                _id=id,
                headSha=pr_obj['head']['sha'],
                url=pr_obj['html_url'],
                type='pr',
                timestamp=datetime.fromisoformat(pr_obj['created_at']),
                author=author,
                title=pr_obj['title'],
                status=pr_obj['state'],
                commits=[],
                description=pr_obj['body'],
            )
            self._db_client.insert_pull_request(pr)
            return asdict(pr)

    def handle_pr_payload(self, pr_payload: dict):
        action = pr_payload['action']
        pr_obj = pr_payload['pull_request']

        if pr_obj['draft']:
            return
        
        if pr_obj['base']['repo']['full_name'] != 'iree-org/wave' or \
            pr_obj['base']['ref'] != 'main':
            return

        is_merge = pr_obj['merged']
        entry_id = f"{pr_obj['id']}_merge" if is_merge else str(pr_obj['id'])

        try:
            mod = self._db_client.find_modification_by_id('merge' if is_merge else 'pr', entry_id)
            print('Found modification in database')
            has_changed = pr_obj['commits'] != mod.commits
        except:
            print('Modification not found in database')
            has_changed = True
        
        pr_entry = self._save_pr(entry_id, pr_obj)

        if has_changed:
            print(f'Pull Request {pr_obj["html_url"]} triggering workflow on {action}')
            head_repo_name = pr_obj['head']['repo']['full_name']
            head_branch = pr_obj['head']['ref']
            self.trigger_workflow(head_repo_name, head_branch, { 'trigger': pr_entry })
