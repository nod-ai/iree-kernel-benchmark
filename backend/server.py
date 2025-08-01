from flask import Flask, jsonify, request
from flask_cors import CORS
from dataclass_wizard import fromdict, asdict

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


@app.route("/artifact/<run_id>")
def get_artifact_by_run_id(run_id):
    new_kernels = load_artifact_kernels(directory_client, f"{run_id}/benchmark-results")
    if run_id == "baseline":
        return jsonify(new_kernels)

    old_kernels = load_artifact_kernels(directory_client, "baseline/benchmark-results")
    filled_kernels = fill_new_kernels(old_kernels, new_kernels)

    return jsonify(filled_kernels)


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


@app.route("/pull_requests/rebase", methods=["POST"])
def rebase_prs():
    rebase_all()
    modifications = db_client.find_all_modifications()
    return jsonify([asdict(modification) for modification in modifications])


if __name__ == "__main__":
    app.run(port=3000)
