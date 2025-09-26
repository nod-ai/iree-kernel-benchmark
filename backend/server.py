import logging
from backend.github_utils import trigger_workflow_dispatch, create_gist, get_repo
from backend.runs import RunType, get_artifact_parser
from backend.runs.run_utils import get_run_by_blob_name
from backend.runs.tracker import get_run_tracker
from backend.storage.rebase import rebase_all
from backend.storage.types import *
from backend.webhook.wave_update import WaveUpdateListener
from backend.storage.auth import get_blob_client

from uuid import uuid4
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from dataclass_wizard import fromdict, asdict
from functools import wraps
import jwt
from datetime import datetime, timezone, timedelta
import os
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv


directory_client = get_blob_client()
logging.getLogger("backend").setLevel(logging.DEBUG)

app = Flask(__name__)
CORS(app, supports_credentials=True)

load_dotenv()
app.config["SECRET_KEY"] = os.getenv("PEM_FILE")
app.config["PASSWORD_HASH"] = os.getenv("PASSWORD_HASH")


@app.route("/")
def home():
    return jsonify({"message": "Flask server with CORS is running on port 3000."})


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")

        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            if token.startswith("Bearer "):
                token = token[7:]

            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Token is invalid"}), 401

        return f(*args, **kwargs)

    return decorated


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    password = data.get("password")

    if not password:
        return jsonify({"message": "Password is required"}), 400

    if check_password_hash(app.config["PASSWORD_HASH"], password):
        token = jwt.encode(
            {"exp": datetime.now(timezone.utc) + timedelta(minutes=30)},
            app.config["SECRET_KEY"],
            algorithm="HS256",
        )

        response = make_response(jsonify({"message": "Login successful"}))
        response.set_cookie(
            "auth_token",
            token,
            max_age=30 * 60,
            httponly=True,
            secure=True,
            samesite="Lax",
        )
        return response

    return jsonify({"message": "Invalid password"}), 401


@app.route("/auth/verify", methods=["GET"])
def verify():
    token = request.cookies.get("auth_token")

    if not token:
        return jsonify({"authenticated": False}), 200

    try:
        jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return jsonify({"authenticated": True}), 200
    except:
        return jsonify({"authenticated": False}), 200


@app.route("/auth/logout", methods=["POST"])
def logout():
    response = make_response(jsonify({"message": "Logout successful"}))
    response.set_cookie("auth_token", "", expires=0)
    return response


@app.route("/pull_requests")
def get_pull_requests():
    modifications = RepoPullRequestDb.find_all()
    return jsonify([asdict(modification) for modification in modifications])


@app.route("/runs")
def get_all_runs():
    runs = BenchmarkRunDb.find_all()
    return jsonify([asdict(run) for run in runs])


@app.route("/kernels")
def get_all_kernels():
    kernels = KernelDb.find_all()
    return jsonify([asdict(k) for k in kernels])


@app.route("/performances")
def get_all_perfs():
    perfs = PerformanceDb.find_all()
    return jsonify([asdict(perf) for perf in perfs])


@app.route("/artifact/<blob_name>")
def get_artifact_by_run_id(blob_name):
    new_kernels = get_artifact_parser(RunType.BENCHMARK).load_data(blob_name)
    if new_kernels:
        return jsonify(new_kernels)
    else:
        return "Failed to gather artifact data", 500


@app.route("/workflow/trigger", methods=["POST"])
def trigger_workflow():
    pr_data = request.get_json()
    wave_client = WaveUpdateListener()
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
    modifications = RepoPullRequestDb.find_all()
    performances = PerformanceDb.find_all()
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

    kernels = KernelDb.find_all()
    tuning_kernels = [asdict(k) for k in kernels if k.id in kernel_ids]

    tuning_request_id = uuid4()
    tuning_upload = create_gist(
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
    return jsonify(TuningConfigDb.find_all())


def serve_backend(port=3000):
    app.run("0.0.0.0", port=port)


if __name__ == "__main__":
    serve_backend()
