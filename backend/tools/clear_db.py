from backend.storage.directory import DirectoryClient
from backend.legacy.db import DatabaseClient
from dotenv import load_dotenv
import os

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "benchmarkcache"
directory_client = DirectoryClient(connection_string, container_name)
db_client = DatabaseClient(connection_string)
db_client.clear_all_runs()
db_client.clear_all_repos()
