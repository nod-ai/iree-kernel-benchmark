from github import Github, Auth, Repository
from globals import TESTING_MODE
from storage import DirectoryClient, DatabaseClient
import os
from dotenv import load_dotenv

load_dotenv()

# Azure

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "benchmarkcache"

db_client = DatabaseClient(connection_string)
directory_client = DirectoryClient(connection_string, container_name)


def get_azure_clients():
    return db_client, directory_client


# GitHub

INSTALLATION_IDS = {
    "wave": int(os.getenv("WAVE_INSTALLATION_ID")),
    "bench": int(os.getenv("BENCH_INSTALLATION_ID")),
    "test": int(os.getenv("TEST_INSTALLATION_ID")),
}

REPO_NAMES = {
    "wave": "iree-org/wave",
    "bench": "nod-ai/iree-kernel-benchmark",
    "test": "suryajasper/github-api-test",
}

APP_ID = int(os.getenv("APP_ID"))
PEM_KEY_PATH = str(os.getenv("PEM_FILE"))

with open(PEM_KEY_PATH, "r") as pem_file:
    signing_key = pem_file.read()

gh_auth = Auth.AppAuth(APP_ID, signing_key)

inst_auths: dict[str, Auth.Auth] = {}
gh_wrappers: dict[str, Github] = {}

for repo_id, installation_id in INSTALLATION_IDS.items():
    inst_auths[repo_id] = gh_auth.get_installation_auth(installation_id)
    gh_wrappers[repo_id] = Github(auth=inst_auths[repo_id])


def get_repo(repo_id: str) -> Repository.Repository:
    return gh_wrappers[repo_id].get_repo(REPO_NAMES[repo_id])


def get_github_token(repo_id: str) -> str:
    return inst_auths[repo_id].token


def get_gist_token() -> str:
    return os.getenv("GIST_TOKEN")
