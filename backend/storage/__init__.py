from .directory import DirectoryClient
from .db import DatabaseClient
from dotenv import load_dotenv
import os

def get_azure_clients():
    load_dotenv()
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = 'benchmarkcache'

    directory_client = DirectoryClient(connection_string, container_name)
    db_client = DatabaseClient(connection_string)

    return directory_client, db_client
