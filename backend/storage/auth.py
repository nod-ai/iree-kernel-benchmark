from backend.storage import DirectoryClient
import os
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "benchmarkcache"
directory_client = DirectoryClient(connection_string, container_name)


def get_blob_client():
    return directory_client
