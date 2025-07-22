from .directory import DirectoryClient
from .db import DatabaseClient
from storage.conversion import convert_prs_from_github
import json
from dotenv import load_dotenv
import os

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = 'benchmarkcache'

directory_client = DirectoryClient(connection_string, container_name)
db_client = DatabaseClient(connection_string)

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

with open('test/pull_requests.json', 'r') as file:
    github_prs = json.load(file)

modifications = convert_prs_from_github(github_prs)
print(modifications)