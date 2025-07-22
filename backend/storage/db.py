from azure.data.tables import TableServiceClient, TableEntity
from .types import *
from dataclass_wizard import fromdict, asdict
from typing import Literal, Optional
import uuid
import json

class DatabaseClient:
    def __init__(self, connection_string: str):
        self._service_client = TableServiceClient.from_connection_string(connection_string)
        self._run_tb = self._service_client.create_table_if_not_exists("runresults")
        self._repo_tb = self._service_client.create_table_if_not_exists("repomodifications")

    def insert_run(self, entry: RunResultEntry) -> str:
        entry_id = str(uuid.uuid4())
        entity = asdict(entry)
        entity.update({
            "PartitionKey": "run",
            "RowKey": entry_id,
        })
        self._run_tb.create_entity(entity)
        return entry_id

    def delete_run(self, entry_id: str):
        self._run_tb.delete_entity(partition_key="run", row_key=entry_id)

    def find_run_by_id(self, entry_id: str) -> Optional[dict]:
        # Search all partitions (inefficient, but required if only RowKey is known)
        entities = self._run_tb.list_entities()
        for entity in entities:
            if entity["RowKey"] == entry_id:
                return entity
        return None

    def find_run_by_trigger_id(self, trigger_id: str) -> list[dict]:
        return list(self._run_tb.query_entities(f"triggerId eq '{trigger_id}'"))

    def find_all_runs(self) -> list[dict]:
        entities = list(self._run_tb.list_entities())
        return sorted(entities, key=lambda x: x["timestamp"], reverse=True)
    
    def insert_pull_request(self, pr: RepoPullRequest) -> str:
        entity = {
            "PartitionKey": "pull",
            "RowKey": pr._id,
            "type": pr.type,
            "timestamp": pr.timestamp,
            "author": json.dumps(asdict(pr.author)),
            "changeStats": json.dumps(pr.changeStats),
            "title": pr.title,
            "description": pr.description or "",
            "status": pr.status,
            "commits": json.dumps([asdict(c) for c in pr.commits])
        }
        self._repo_tb.create_entity(entity)
        return pr._id

    def update_pull_request(self, prId: str, **kwargs):
        entity = self._repo_tb.get_entity(partition_key="pull", row_key=prId)
        for key, value in kwargs.items():
            if key in {"author", "changeStats", "commits"}:
                if key == "author":
                    entity[key] = json.dumps(asdict(value))
                elif key == "commits":
                    entity[key] = json.dumps([asdict(c) for c in value])
                else:
                    entity[key] = json.dumps(value)
            else:
                entity[key] = value
        self._repo_tb.update_entity(entity, mode="MERGE")

    def delete_pull_request(self, prId: str) -> str:
        self._repo_tb.delete_entity(partition_key="pull", row_key=prId)
        return prId

    def insert_merge(self, pr: RepoMerge) -> str:
        entity = {
            "PartitionKey": "merge",
            "RowKey": pr._id,
            "type": pr.type,
            "timestamp": pr.timestamp,
            "author": json.dumps(asdict(pr.author)),
            "changeStats": json.dumps(pr.changeStats),
            "prId": pr.prId
        }
        self._repo_tb.create_entity(entity)
        return pr._id

    def update_merge(self, mergeId: str, **kwargs):
        entity = self._repo_tb.get_entity(partition_key="merge", row_key=mergeId)
        for key, value in kwargs.items():
            if key in {"author", "changeStats"}:
                entity[key] = json.dumps(asdict(value) if key == "author" else value)
            else:
                entity[key] = value
        self._repo_tb.update_entity(entity, mode="MERGE")

    def delete_merge(self, mergeId: str) -> str:
        self._repo_tb.delete_entity(partition_key="merge", row_key=mergeId)
        return mergeId

    def find_all_modifications(self) -> list[RepoModification]:
        entities = list(self._repo_tb.list_entities())
        entities_sorted = sorted(entities, key=lambda x: x["timestamp"], reverse=True)
        modifications = []
        for entity in entities_sorted:
            entity_dict = entity.copy()
            entity_dict['author'] = json.loads(entity_dict['author'])
            entity_dict['changeStats'] = json.loads(entity_dict['changeStats'])
            if entity_dict['type'] == 'pr':
                entity_dict['commits'] = json.loads(entity_dict['commits'])
                modifications.append(fromdict(RepoPullRequest, entity_dict))
            else:
                modifications.append(fromdict(RepoMerge, entity_dict))
    
    
    def clear_all_runs(self):
        entities = self._run_tb.list_entities()
        for entity in entities:
            self._run_tb.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])

    def clear_all_repos(self):
        entities = self._repo_tb.list_entities()
        for entity in entities:
            self._repo_tb.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])