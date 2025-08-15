from auth import get_repo, get_azure_clients
from storage.runs import (
    update_runs,
    update_run_mappings,
    update_change_stats,
    update_artifacts,
    update_tuning_runs,
)
import time
import traceback

UPDATE_RUNS_INTERVAL = 10  # seconds
UPDATE_MAPPINGS_INTERVAL = 15  # seconds
UPDATE_ARTIFACTS_INTERVAL = 15  # seconds
UPDATE_TUNING_RUNS_INTERVAL = 20  # seconds
UPDATE_CHANGE_STATS_INTERVAL = 20  # seconds

db_client, dir_client = get_azure_clients()
repo = get_repo("bench")


def serve_event_loop():
    last_run_update_time = 0
    last_tuning_run_update_time = 0
    last_mapping_update_time = 0
    last_change_stats_update_time = 0
    last_artifacts_update_time = 0

    while True:
        now = time.time()

        if now - last_run_update_time >= UPDATE_RUNS_INTERVAL:
            try:
                update_runs(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_runs:")
                traceback.print_exc()
            last_run_update_time = now

        if now - last_tuning_run_update_time >= UPDATE_TUNING_RUNS_INTERVAL:
            try:
                update_tuning_runs()
            except Exception:
                print("Exception occurred in update_tuning_runs:")
                traceback.print_exc()
            last_tuning_run_update_time = now

        if now - last_mapping_update_time >= UPDATE_MAPPINGS_INTERVAL:
            try:
                update_run_mappings(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_mappings:")
                traceback.print_exc()
            last_mapping_update_time = now

        if now - last_artifacts_update_time >= UPDATE_ARTIFACTS_INTERVAL:
            try:
                update_artifacts(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_artifacts:")
                traceback.print_exc()
            last_artifacts_update_time = now

        if now - last_change_stats_update_time >= UPDATE_CHANGE_STATS_INTERVAL:
            try:
                update_change_stats(repo, db_client, dir_client)
            except Exception:
                print("Exception occurred in update_change_stats:")
                traceback.print_exc()
            last_change_stats_update_time = now

        time.sleep(1)


if __name__ == "__main__":
    serve_event_loop()
