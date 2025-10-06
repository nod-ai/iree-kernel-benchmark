import logging
import traceback
from backend.github_utils import create_gist, get_repo
from backend.github_utils.gist import load_gist_by_id
from backend.runs import RunType, get_artifact_parser
from backend.runs.run_utils import find_incomplete_runs, get_run_by_blob_name
from backend.runs.tracker import get_run_tracker
from backend.runs.workflows import trigger_bench_workflow
from backend.storage.rebase import rebase_all, rebase_pull_requests
from backend.storage.types import *
from backend.storage.utils import test_logger
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

app = Flask(__name__)
CORS(app, supports_credentials=True)

load_dotenv()
app.config["SECRET_KEY"] = os.getenv("PEM_FILE")
app.config["PASSWORD_HASH"] = os.getenv("PASSWORD_HASH")

logger = logging.getLogger(__name__)


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
    runs = WorkflowRunDb.find_all({"type": RunType.BENCHMARK.name})
    return jsonify([asdict(run) for run in runs])


@app.route("/performances")
def get_all_perfs():
    perfs = WorkflowRunDb.find_all({"type": RunType.E2E.name})
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
    response_data = request.get_json()

    pr_data = response_data["pr"]
    config_data = response_data["config"]

    wave_client = WaveUpdateListener()
    trigger_success = wave_client.trigger_workflow(
        pr_data["repoName"], pr_data["branchName"], pr_data["mappingId"]
    )

    kernel_selection = config_data["kernelSelection"]
    if kernel_selection["type"] == "specific-tags":
        tags = kernel_selection["tags"]
        bench_kernels = KernelConfigDb.query(
            " or ".join([f"tag eq '{tag}'" for tag in tags])
        )
        logger.info(
            f"Loaded {len(bench_kernels)} kernels for benchmark with {len(tags)} tags"
        )
    else:
        bench_kernels = KernelConfigDb.find_all({"workflow": "all"})
        logger.info(f"Loaded {len(bench_kernels)} quick kernels for benchmark")

    if len(bench_kernels) == 0:
        return "No kernels found", 500

    bench_kernels_json = [asdict(k) for k in bench_kernels]
    problems_gist = create_gist(bench_kernels_json)

    tuned_configs = TuningConfigDb.find_all()
    tuned_configs_json = {c.kernel_name: c.result for c in tuned_configs}
    tuned_gist = create_gist(tuned_configs_json)

    trigger_success = trigger_bench_workflow(
        RunType.BENCHMARK,
        {
            "selected_backend": "all",
            "selected_kernel": "all",
            "problems_url": problems_gist.raw_url,
            "tuned_config_url": tuned_gist.raw_url,
            "pr_repository": pr_data["repoName"],
            "pr_branch": pr_data["branchName"],
            "pr_headsha": pr_data["mappingId"],
        },
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
    # rebase_all()
    rebase_pull_requests()
    modifications = RepoPullRequestDb.find_all()
    performances = []
    # performances = WorkflowRunDb.find_all({"type": RunType.E2E.name})
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

    kernels = KernelConfigDb.find_all()
    tuning_kernels = [asdict(k) for k in kernels if k._id in kernel_ids]

    tuning_request_id = uuid4()
    tuning_upload = create_gist(
        tuning_kernels,
        filename=f"tuning-request-{tuning_request_id}",
        description=f"Tuning configuration for {len(tuning_kernels)} kernels",
    )

    if not tuning_upload:
        return "Failed to upload config to gist", 500

    dispatch_success = trigger_bench_workflow(
        RunType.TUNING,
        {
            "problems_url": tuning_upload.raw_url,
            "identifier": tuning_upload.gist_id,
        },
    )

    if dispatch_success:
        return "Success", 200
    else:
        return "Failed to dispatch workflow", 500


@app.route("/tune/results", methods=["GET"])
def get_tuned_results():
    tuned_results = TuningConfigDb.find_all()
    return jsonify([asdict(t) for t in tuned_results])


@app.route("/tune/runs", methods=["GET"])
def get_tuning_runs():
    runs = find_incomplete_runs(RunType.TUNING)

    tuning_kernels = []
    for run in runs:
        if not run.mappingId:
            continue
        run_kernels = load_gist_by_id(run.mappingId)
        if run_kernels:
            tuning_kernels.extend(run_kernels)

    return jsonify(
        {
            "runs": [asdict(r) for r in runs],
            "kernels": tuning_kernels,
        }
    )


@app.route("/change_stats", methods=["GET"])
def get_all_change_stats():
    """Get all kernel types from the database."""
    change_stats = ChangeStatDb.find_all()
    return jsonify([asdict(cs) for cs in change_stats])


@app.route("/change_stats/<run_id>", methods=["GET"])
def get_change_stat_by_run_id(run_id):
    """Get all kernel types from the database."""
    change_stats = ChangeStatDb.find_all({"runId": str(run_id)})
    if len(change_stats) == 0:
        return "Failed to find change stats", 500
    return jsonify(asdict(change_stats[0]))


@app.route("/kernel_types", methods=["GET"])
def get_all_kernel_types():
    """Get all kernel types from the database."""
    kernel_types = KernelTypeDb.find_all()
    return jsonify([asdict(kt) for kt in kernel_types])


@app.route("/kernel_types", methods=["POST"])
# @token_required
def add_kernel_type():
    """Add a new kernel type to the database."""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["_id", "name", "displayName", "attributes"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create kernel type from the request data
        kernel_type = fromdict(KernelTypeDefinition, data)

        # Check if kernel type with this ID already exists
        existing = KernelTypeDb.find_by_id(kernel_type._id)
        if existing:
            return jsonify({"error": "Kernel type with this ID already exists"}), 409

        # Save to database
        success = KernelTypeDb.upsert(kernel_type)

        if success:
            return jsonify(asdict(kernel_type)), 201
        else:
            return jsonify({"error": "Failed to save kernel type"}), 500

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400


@app.route("/kernel_types/<kernel_type_id>", methods=["PUT"])
# @token_required
def update_kernel_type(kernel_type_id):
    """Update an existing kernel type."""
    try:
        data = request.get_json()

        # Check if kernel type exists
        existing = KernelTypeDb.find_by_id(kernel_type_id)
        if not existing:
            return jsonify({"error": "Kernel type not found"}), 404

        # Update the kernel type
        updated = KernelTypeDb.update_by_id(kernel_type_id, data)

        if updated:
            return jsonify(asdict(updated)), 200
        else:
            return jsonify({"error": "Failed to update kernel type"}), 500

    except Exception as e:
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400


@app.route("/kernel_types/<kernel_type_id>", methods=["DELETE"])
# @token_required
def remove_kernel_type(kernel_type_id):
    """Remove a kernel type from the database."""
    try:
        # Check if kernel type exists
        existing = KernelTypeDb.find_by_id(kernel_type_id)
        if not existing:
            return jsonify({"error": "Kernel type not found"}), 404

        # Delete the kernel type
        success = KernelTypeDb.delete_by_id(kernel_type_id)

        if success:
            return jsonify({"message": "Kernel type deleted successfully"}), 200
        else:
            return jsonify({"error": "Failed to delete kernel type"}), 500

    except Exception as e:
        return jsonify({"error": f"Error deleting kernel type: {str(e)}"}), 500


@app.route("/kernels", methods=["GET"])
def get_all_kernels():
    """Get all kernel configurations from the database."""
    kernels = KernelConfigDb.find_all()
    return jsonify([asdict(k) for k in kernels])


@app.route("/kernels", methods=["POST"])
# @token_required
def add_kernels():
    """Add multiple new kernel configurations to the database."""
    try:
        data = request.get_json()

        # Expect either a single kernel config or a list of kernel configs
        if not isinstance(data, list):
            kernel_configs = [data]
        else:
            kernel_configs = data

        if not kernel_configs:
            return jsonify({"error": "No kernel configurations provided"}), 400

        created_kernels = []

        for kernel_data in kernel_configs:
            # Validate required fields (excluding _id as it can be auto-generated)
            required_fields = [
                "name",
                "kernelType",
                "tag",
                "machines",
                "workflow",
                "problem",
            ]
            for field in required_fields:
                if field not in kernel_data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Generate unique ID if not provided
            if "_id" not in kernel_data or not kernel_data["_id"]:
                kernel_data["_id"] = str(uuid4())

            # Create kernel config from the request data
            kernel_config = fromdict(KernelConfig, kernel_data)
            created_kernels.append(kernel_config)

        # Save all kernel configs to database
        success = KernelConfigDb.upsert_many(created_kernels)

        if success:
            return jsonify([asdict(k) for k in created_kernels]), 201
        else:
            return jsonify({"error": "Failed to save kernel configurations"}), 500

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400


@app.route("/kernels/<kernel_id>", methods=["PUT"])
# @token_required
def update_kernel(kernel_id):
    """Update an existing kernel configuration."""
    try:
        data = request.get_json()

        # Check if kernel config exists
        existing = KernelConfigDb.find_by_id(kernel_id)
        if not existing:
            return jsonify({"error": "Kernel configuration not found"}), 404

        # Update the kernel config
        updated = KernelConfigDb.update_by_id(kernel_id, data)

        if updated:
            return jsonify(asdict(updated)), 200
        else:
            return jsonify({"error": "Failed to update kernel configuration"}), 500

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400


@app.route("/kernels/batch", methods=["PUT"])
# @token_required
def update_kernels_batch():
    """Update multiple kernel configurations using batched transactions."""
    try:
        data = request.get_json()

        # Expect a list of update dictionaries
        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of update objects"}), 400

        if not data:
            return jsonify({"error": "No kernel updates provided"}), 400

        # Validate that each update has an _id field
        for i, update_dict in enumerate(data):
            if not isinstance(update_dict, dict):
                return jsonify({"error": f"Update at index {i} must be an object"}), 400
            if "_id" not in update_dict:
                return (
                    jsonify(
                        {"error": f"Update at index {i} missing required '_id' field"}
                    ),
                    400,
                )

        # Perform batched update
        success = KernelConfigDb.update_many(data)

        if success:
            return (
                jsonify(
                    {
                        "message": f"Successfully updated {len(data)} kernel configurations",
                        "updated_count": len(data),
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "Failed to update kernel configurations"}), 500

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400


@app.route("/kernels", methods=["DELETE"])
# @token_required
def remove_kernels():
    """Remove multiple kernel configurations from the database."""
    try:
        data = request.get_json()

        # Expect a list of kernel IDs to delete
        if not data or "ids" not in data:
            return jsonify({"error": "Missing 'ids' field in request body"}), 400

        kernel_ids = data["ids"]
        if not isinstance(kernel_ids, list):
            return jsonify({"error": "'ids' must be a list"}), 400

        if not kernel_ids:
            return jsonify({"error": "No kernel IDs provided"}), 400

        success = KernelConfigDb.delete_many_by_ids(kernel_ids)

        # Return appropriate status code based on results
        if success:
            return f"Successfully deleted {len(kernel_ids)} kernels", 200
        else:
            return "Failed to delete one or more kernels", 400

    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Error deleting kernel configurations: {str(e)}", 500


def serve_backend(port=3000):
    app.run("0.0.0.0", port=port)


if __name__ == "__main__":
    serve_backend()
