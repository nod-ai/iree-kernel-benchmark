
from auth import get_access_token
from github import Github, Auth
from globals import TESTING_MODE
from storage.runs import update_runs, update_change_stats, update_artifacts
from storage.db import DatabaseClient
from storage.directory import DirectoryClient
import os
from dotenv import load_dotenv
import time
import traceback

UPDATE_RUNS_INTERVAL = 10         # seconds
UPDATE_ARTIFACTS_INTERVAL = 15 # seconds
UPDATE_CHANGE_STATS_INTERVAL = 20 # seconds

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = 'benchmarkcache'

db_client = DatabaseClient(connection_string)
dir_client = DirectoryClient(connection_string, container_name)

auth = Auth.Token(get_access_token('TEST' if TESTING_MODE else 'BENCH'))
g = Github(auth=auth)
repo = g.get_repo('suryajasper/github-api-test' if TESTING_MODE else 'nod-ai/iree-kernel-benchmark')

def run_forever():
    last_run_update_time = 0
    last_change_stats_update_time = 0
    last_artifacts_update_time = 0

    while True:
        now = time.time()

        if now - last_run_update_time >= UPDATE_RUNS_INTERVAL:
            try:
                print("Running update_runs...")
                update_runs(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_runs:")
                traceback.print_exc()
            last_run_update_time = now

        if now - last_artifacts_update_time >= UPDATE_ARTIFACTS_INTERVAL:
            try:
                print("Running update_artifacts...")
                update_artifacts(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_artifacts:")
                traceback.print_exc()
            last_artifacts_update_time = now
        
        if now - last_change_stats_update_time >= UPDATE_CHANGE_STATS_INTERVAL:
            try:
                print("Running update_change_stats...")
                update_change_stats(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_change_stats:")
                traceback.print_exc()
            last_change_stats_update_time = now

        time.sleep(1)

if __name__ == "__main__":
    run_forever()
