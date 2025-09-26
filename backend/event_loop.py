from backend.github_utils import get_repo
from backend.runs.manager import RunManager
import time
import traceback

UPDATE_RUNS_INTERVAL = 10  # seconds

repo = get_repo("bench")


def serve_event_loop():
    last_run_update_time = 0
    run_manager = RunManager()

    while True:
        now = time.time()

        if now - last_run_update_time >= UPDATE_RUNS_INTERVAL:
            try:
                run_manager.update_runs()
            except Exception:
                print("Exception occurred in update_runs:")
                traceback.print_exc()
            last_run_update_time = now

        time.sleep(1)


if __name__ == "__main__":
    serve_event_loop()
