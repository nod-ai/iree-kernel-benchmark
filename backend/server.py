from uuid import uuid4
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from dataclass_wizard import fromdict, asdict

from storage.runs import trigger_workflow_dispatch, upload_json_to_gist
from storage.rebase import rebase_all
from webhook.wave_update import WaveUpdateListener
from auth import get_azure_clients, get_repo
from storage.artifacts import (
    load_artifact_kernels,
    fill_new_kernels,
)


db_client, directory_client = get_azure_clients()

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"message": "Flask server with CORS is running on port 3000."})


@app.route("/pull_requests")
def get_pull_requests():
    modifications = db_client.find_all_modifications()
    return jsonify([asdict(modification) for modification in modifications])


@app.route("/runs")
def get_all_runs():
    runs = db_client.find_all_runs()
    return jsonify([asdict(run) for run in runs])


@app.route("/kernels")
def get_all_kernels():
    kernels = db_client.find_all_kernels()
    return jsonify([asdict(k) for k in kernels])


@app.route("/performances")
def get_all_perfs():
    perfs = db_client.find_all_performances()
    return jsonify([asdict(perf) for perf in perfs])


@app.route("/artifact/<run_id>")
def get_artifact_by_run_id(run_id):
    new_kernels = load_artifact_kernels(directory_client, f"{run_id}/benchmark-results")
    if run_id == "baseline":
        return jsonify(new_kernels)

    print(any([k["backend"] == "wavenew" for k in new_kernels]))

    # old_kernels = load_artifact_kernels(directory_client, "baseline/benchmark-results")
    # filled_kernels = fill_new_kernels(old_kernels, new_kernels)
    # print(any([k["backend"] == "wavenew" for k in filled_kernels]))

    return jsonify(new_kernels)


@app.route("/workflow/trigger", methods=["POST"])
def trigger_workflow():
    pr_data = request.get_json()
    wave_client = WaveUpdateListener(db_client, directory_client)
    trigger_success = wave_client.trigger_workflow(
        pr_data["repoName"], pr_data["branchName"], pr_data["headSha"]
    )
    if trigger_success:
        return "Success", 200
    else:
        return "Failure", 500


@app.route("/workflow/cancel", methods=["POST"])
def cancel_workflow():
    payload = request.get_json()
    run_id = int(payload["runId"])
    cancel_success = get_repo("bench").get_workflow_run(run_id).cancel()
    if cancel_success:
        return "Success", 200
    else:
        return "Failure", 500


@app.route("/rebase", methods=["POST"])
def rebase_prs():
    rebase_all()
    modifications = db_client.find_all_modifications()
    performances = db_client.find_all_performances()
    return jsonify(
        {
            "modifications": [asdict(modification) for modification in modifications],
            "performances": [asdict(perf) for perf in performances],
        }
    )


@app.route("/tune", methods=["POST"])
def tune_kernels():
    payload = request.get_json()
    kernel_ids = [str(id) for id in payload["kernel_ids"]]

    kernels = db_client.find_all_kernels()
    tuning_kernels = [asdict(k) for k in kernels if k.id in kernel_ids]

    tuning_request_id = uuid4()
    tuning_upload = upload_json_to_gist(
        tuning_kernels,
        filename=f"tuning-request-{tuning_request_id}",
        description=f"Tuning configuration for {len(tuning_kernels)} kernels",
    )

    if not tuning_upload["success"]:
        return "Failed to upload config to gist", 500

    json_url = tuning_upload["raw_url"]

    dispatch_success = trigger_workflow_dispatch(
        "bench",
        "kernel-dashboard",
        "tune_kernels.yml",
        {
            "config_url": json_url,
            "num_trials": "20",
            "iterations": "1",
        },
    )

    if dispatch_success:
        return "Success", 200
    else:
        return "Failed to dispatch workflow", 500


@app.route("/tune/results", methods=["GET"])
def get_tuned_results():
    return jsonify(db_client.find_all_tuning_configs())


def serve_backend(port=3000):
    app.run("0.0.0.0", port=port)


if __name__ == "__main__":
    serve_backend()
