from auth import get_access_token
from github import Github, Auth, PullRequest, Repository
from flask import Flask, jsonify
from flask_cors import CORS
from dataclass_wizard import fromdict, asdict

from storage import get_azure_clients
from storage.artifacts import fetch_latest_artifact, load_artifact_kernels
from storage.conversion import convert_prs_from_github
import json


directory_client, db_client = get_azure_clients()

app = Flask(__name__)
CORS(app)

def get_repo(repo_name: str) -> Repository.Repository:
    auth = Auth.Token(get_access_token())
    g = Github(auth=auth)
    return g.get_repo(repo_name)

@app.route('/')
def home():
    return jsonify({"message": "Flask server with CORS is running on port 3000."})

@app.route('/pull_requests')
def get_pull_requests():
    modifications = db_client.find_all_modifications()
    return jsonify([asdict(modification) for modification in modifications])

@app.route('/runs')
def get_all_runs():
    runs = db_client.find_all_runs()
    return jsonify([asdict(run) for run in runs])

@app.route('/artifact')
def get_latest_artifact():
    print('hi')
    # artifact = fetch_latest_artifact(directory_client, db_client)
    # return jsonify(artifact.kernels)
    return jsonify(
        load_artifact_kernels(directory_client, '2025-07-23T17:30:22.817728+00:00/benchmark-results')
    )

if __name__ == '__main__':
    app.run(port=3000)
