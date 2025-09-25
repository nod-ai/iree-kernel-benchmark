from backend.storage import DirectoryClient, DatabaseClient
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
