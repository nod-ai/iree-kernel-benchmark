from azure.data.tables import TableServiceClient, TableEntity
from .types import *
from dataclasses import is_dataclass
from dataclass_wizard import fromdict, asdict
from typing import Literal, Optional, Any
import uuid
import json
from datetime import datetime, timezone

def flatten_entry(entry: Any, row_key: str = None, partition: str = None, indent = 4) -> dict[str, Any]:
    new_obj = {}
    if row_key:
        new_obj['RowKey'] = row_key
    if partition:
        new_obj['PartitionKey'] = partition
    
    if is_dataclass(entry):
        entry = asdict(entry)
    
    for key, value in entry.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=indent)
        new_obj[key] = value
    
    return new_obj

class DatabaseClient:
    def __init__(self, connection_string: str):
        self._service_client = TableServiceClient.from_connection_string(connection_string)
        self._run_tb = self._service_client.create_table_if_not_exists("runresults")
        self._repo_tb = self._service_client.create_table_if_not_exists("repomodifications")

    def insert_run(self, run: BenchmarkRun) -> str:
        run_obj = flatten_entry(run, row_key=run._id, partition="run")
        self._run_tb.upsert_entity(run_obj)
        return run._id
    
    def update_run(self, run_id: str, **kwargs):
        entity = self._run_tb.get_entity(partition_key="run", row_key=run_id)
        entity.update(kwargs)
        entity = flatten_entry(entity)
        self._run_tb.update_entity(entity)

    def delete_run(self, run_id: str):
        self._run_tb.delete_entity(partition_key="run", row_key=run_id)

    def find_run_by_id(self, run_id: str) -> dict:
        return self._run_tb.get_entity(partition_key="run", row_key=run_id)

    def query_runs(self, query: str) -> list[BenchmarkRun]:
        entities = list(self._run_tb.query_entities(query))
        runs = []
        for entity in entities:
            entity['steps'] = json.loads(entity['steps'])
            entity['changeStats'] = json.loads(entity['changeStats'])
            runs.append(fromdict(BenchmarkRun, entity))
        return runs

    def find_all_runs(self) -> list[BenchmarkRun]:
        entities = list(self._run_tb.list_entities())
        runs = []
        for entity in entities:
            entity['steps'] = json.loads(entity['steps'])
            entity['changeStats'] = json.loads(entity['changeStats'])
            runs.append(fromdict(BenchmarkRun, entity))
        return runs
    
    def insert_pull_request(self, pr: RepoPullRequest) -> str:
        entity = flatten_entry(pr, row_key=pr._id, partition="pull")
        self._repo_tb.upsert_entity(entity)
        return pr._id

    def update_pull_request(self, prId: str, **kwargs):
        entity = self._repo_tb.get_entity(partition_key="pull", row_key=prId)
        entity.update(flatten_entry(kwargs))
        self._repo_tb.update_entity(entity)

    def delete_pull_request(self, prId: str) -> str:
        self._repo_tb.delete_entity(partition_key="pull", row_key=prId)
        return prId

    def insert_merge(self, pr: RepoMerge) -> str:
        entity = flatten_entry(pr, row_key=pr._id, partition="merge")
        self._repo_tb.upsert_entity(entity)
        return pr._id

    def update_merge(self, mergeId: str, **kwargs):
        entity = self._repo_tb.get_entity(partition_key="merge", row_key=mergeId)
        entity.update(flatten_entry(kwargs))
        self._repo_tb.update_entity(entity)

    def delete_merge(self, mergeId: str) -> str:
        self._repo_tb.delete_entity(partition_key="merge", row_key=mergeId)
        return mergeId
    
    def query_modifications(self, query: str) -> list[dict]:
        return list(self._repo_tb.query_entities(query))

    def find_modification_by_id(self, type: str, mod_id: str) -> RepoPullRequest | RepoMerge:
        entity = self._repo_tb.get_entity(partition_key=type, row_key=mod_id)
        return fromdict(RepoMerge, entity) if type == 'merge' else fromdict(RepoPullRequest, entity)

    def find_all_modifications(self) -> list[RepoModification]:
        entities = list(self._repo_tb.list_entities())
        entities_sorted = sorted(entities, key=lambda x: x["timestamp"], reverse=True)

        modifications = []
        for entity in entities_sorted:
            entity_dict = entity.copy()
            entity_dict['author'] = json.loads(entity_dict['author'])
            if entity_dict['type'] == 'pr':
                entity_dict['commits'] = json.loads(entity_dict['commits'])
                modifications.append(fromdict(RepoPullRequest, entity_dict))
            else:
                modifications.append(fromdict(RepoMerge, entity_dict))
        
        return modifications
    
    def clear_all_runs(self):
        entities = self._run_tb.list_entities()
        for entity in entities:
            self._run_tb.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])

    def clear_all_repos(self):
        entities = self._repo_tb.list_entities()
        for entity in entities:
            self._repo_tb.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])