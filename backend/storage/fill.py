from .directory import DirectoryClient
from .db import DatabaseClient
from .types import *
from .artifacts import save_all_artifact_kernels
from dataclass_wizard import fromdict
from storage.conversion import convert_prs_from_github, random_stats
from storage.artifacts import load_artifact_kernels
from auth import get_access_token
from github import Github, Auth
import json
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import random
from uuid import uuid4
from tqdm import tqdm


def get_random_indices(sample: int, range_: int) -> list:
    if sample > range_:
        raise ValueError("Sample size cannot be greater than the range.")
    return random.sample(range(range_), sample)

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = 'benchmarkcache'

directory_client = DirectoryClient(connection_string, container_name)
db_client = DatabaseClient(connection_string)

auth = Auth.Token(get_access_token('TEST'))
gh_client = Github(auth=auth)
repo = gh_client.get_repo('nod-ai/iree-kernel-benchmark')

db_client.clear_all_repos()
db_client.clear_all_runs()

num_success = 7
num_total = 40

kernel_collections, blob_names = save_all_artifact_kernels(repo, directory_client, limit=num_success)

modifications : list[dict] = []

with open('test/pull_requests.json', 'rb') as file:
    github_json = json.load(file)
    modifications = convert_prs_from_github(github_json)[:num_total]
    for modification in tqdm(modifications, desc="Loading modifications"):
        if modification['type'] == 'pr':
            db_client.insert_pull_request(fromdict(RepoPullRequest, modification))
        elif modification['type'] == 'merge':
            db_client.insert_merge(fromdict(RepoMerge, modification))

date = datetime.now(timezone.utc)

success_indices = get_random_indices(num_success, num_total)

j = 0
for i in tqdm(range(num_total), desc="Generating Fake Data"):
    mod = modifications[i]

    is_success = i in success_indices
    
    if not is_success and random.random() < 0.2:
        continue

    workflow_states = ["requested", "in_progress", "completed", "queued", "pending", "waiting"]
    workflow_conclusions = ["action_required", "cancelled", "failure", "neutral", "skipped", "stale", "timed_out", "startup_failure"]
    job_conclusions = ["failure", "skipped", "cancelled"]

    status = "completed" if is_success else random.choice(workflow_states)
    if is_success:
        workflow_conclusion = "success"
    elif status == "completed":
        workflow_conclusion = random.choice(workflow_conclusions)
    else:
        workflow_conclusion = "null"
    
    steps = []
    num_total_steps = 5

    if status in ['completed', 'in_progress']:
        if workflow_conclusion == 'success':
            num_steps_success = num_total_steps
        else:
            num_steps_success = random.randint(0, num_total_steps)

        for i in range(min(num_steps_success+1, num_total_steps)):
            step_obj = {
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'name': f'Step {i+1}',
                'number': i+1,
                'started_at': datetime.now(timezone.utc).isoformat(),
            }
            if i < num_steps_success:
                step_obj.update({
                    'conclusion': 'success',
                    'status': 'completed'
                })
                steps.append(step_obj)
            else:
                step_obj.update({
                    'conclusion': 'null' if workflow_conclusion == 'null' else random.choice(job_conclusions),
                    'status': random.choice(('in_progress', 'queued')) if workflow_conclusion == 'null' else 'completed'
                })
                steps.append(step_obj)
                break

    blobName = ""
    if is_success:
        blobName = blob_names[j]
        j += 1

    benchmark_id = str(uuid4())

    change_stats = random_stats() if is_success else {}

    run = BenchmarkRun(
        _id=benchmark_id,
        headSha=mod['headSha'],
        status=status,
        conclusion=workflow_conclusion,
        numSteps=num_total_steps,
        steps=steps,
        blobName=blobName,
        timestamp=datetime.now(timezone.utc),
        changeStats=change_stats
    )
    db_client.insert_run(run)

# results = load_artifact(directory_client, '2025-07-22T07:24:55.435392+00:00/benchmark-results')
# with open('./test/results.json', 'w') as file:
#     json.dump(results, file, indent=4)