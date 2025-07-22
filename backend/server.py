from auth import get_access_token
from github import Github, Auth, PullRequest, Repository
from flask import Flask, jsonify
from flask_cors import CORS
from dataclass_wizard import fromdict

from storage.directory import DirectoryClient
import storage.types as dbtypes
from storage.db import DatabaseClient
from storage.conversion import convert_prs_from_github
import json

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
    # repo = get_repo('iree-org/wave')
    # pull_request_pages = repo.get_pulls(state='all')
    # pr_page = pull_request_pages.get_page(0)
    # pr_objs = []
    # for pr in pr_page:
    #     print(pr)
    #     pr_objs.append(pr.raw_data)
    # return jsonify(pr_objs)
    with open('test/pull_requests.json', 'rb') as file:
        github_prs = json.load(file)

    modifications = convert_prs_from_github(github_prs)
    return jsonify(modifications)

@app.route('/dashboard')
def get_latest_artifact():
    repo = get_repo('nod-ai/iree-kernel-benchmark')
    latest_artifact = repo.get_artifacts().get_page(0)[0]
    

if __name__ == '__main__':
    app.run(port=3000)
