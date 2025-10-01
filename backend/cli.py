from auth import get_repo, get_azure_clients
import os

db_client, dir_client = get_azure_clients()


dir_client.rm(f"baseline", recursive=True)
dir_client.upload(f"test/benchmark-results", "baseline")

db_client.delete_runs("conclusion eq 'failure' or conclusion eq 'cancelled'")
