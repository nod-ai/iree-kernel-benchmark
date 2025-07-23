from .directory import DirectoryClient
from .db import DatabaseClient
from .types import RepoMerge, RepoPullRequest
from dataclass_wizard import fromdict
from storage.conversion import convert_prs_from_github
import json
from dotenv import load_dotenv
import os

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = 'benchmarkcache'

directory_client = DirectoryClient(connection_string, container_name)
db_client = DatabaseClient(connection_string)

db_client.clear_all_repos()
# db_client.clear_all_runs()

# date = datetime.now(timezone.utc)
# timestamp_str = date.isoformat()
# timestamp_int = int(date.timestamp())

# directory_client.upload_dir('test/benchmark-results', timestamp_str)
# db_client.insert_run(RunResultEntry(
#     triggerId='awefopijawepofij',
#     blobName=timestamp_str,
#     runType='pr',
#     timestamp=timestamp_int
# ))
# print(db_client.find_all_runs())

with open('test/pull_requests.json', 'rb') as file:
    github_json = json.load(file)
    modifications = convert_prs_from_github(github_json)
    for modification in modifications:
        if modification['type'] == 'pr':
            db_client.insert_pull_request(fromdict(RepoPullRequest, modification))
        elif modification['type'] == 'merge':
            db_client.insert_merge(fromdict(RepoMerge, modification))

# results = load_artifact(directory_client, '2025-07-22T07:24:55.435392+00:00/benchmark-results')
# with open('./test/results.json', 'w') as file:
#     json.dump(results, file, indent=4)