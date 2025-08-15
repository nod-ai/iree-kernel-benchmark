from math import ceil
from azure.data.tables import TableServiceClient, TableEntity, TableClient
from .types import *
from dataclasses import is_dataclass
from dataclass_wizard import fromdict, asdict
from typing import Literal, Optional, Any
from uuid import uuid4
import json
from datetime import datetime, timezone


def flatten_entry(
    entry: Any, row_key: str = None, partition: str = None, indent=4
) -> dict[str, Any]:
    new_obj = {}
    if row_key:
        new_obj["RowKey"] = row_key
    if partition:
        new_obj["PartitionKey"] = partition

    if is_dataclass(entry):
        entry = asdict(entry)

    for key, value in entry.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=indent)
        new_obj[key] = value

    return new_obj


class DatabaseClient:
    def __init__(self, connection_string: str):
        self._service_client = TableServiceClient.from_connection_string(
            connection_string
        )
        self._tuning_run_tb = self._service_client.create_table_if_not_exists(
            "tuningruns"
        )
        self._run_tb = self._service_client.create_table_if_not_exists("runresults")
        self._repo_tb = self._service_client.create_table_if_not_exists(
            "repomodifications"
        )
        self._performance_tb = self._service_client.create_table_if_not_exists(
            "performanceruns"
        )
        self._kernel_tb = self._service_client.create_table_if_not_exists("kernels")
        self._tuning_tb = self._service_client.create_table_if_not_exists(
            "tuningconfigs"
        )

    def _batch_submit(
        self,
        table: TableClient,
        op_name: str,
        entities: list[dict[str, Any]],
        batch_count=50,
    ):
        num_batches = ceil(len(entities) / batch_count)
        for batch_idx in range(num_batches):
            start = batch_idx * batch_count
            end = min(start + batch_count, len(entities))
            batch_ops = [(op_name, entity) for entity in entities[start:end]]
            table.submit_transaction(batch_ops)

    def insert_run(self, run: BenchmarkRun) -> str:
        run_obj = flatten_entry(run, row_key=run._id, partition="run")
        self._run_tb.upsert_entity(run_obj)
        return run._id

    def update_run(self, run_id: str, update_dict: dict[str, Any]):
        entity = self._run_tb.get_entity(partition_key="run", row_key=run_id)
        entity.update(update_dict)
        entity = flatten_entry(entity)
        self._run_tb.update_entity(entity)

    def delete_run(self, run_id: str):
        self._run_tb.delete_entity(partition_key="run", row_key=run_id)

    def delete_runs(self, query: str):
        queried_runs = self._run_tb.query_entities(query)
        for run in queried_runs:
            self._run_tb.delete_entity(run["PartitionKey"], run["RowKey"])

    def find_run_by_id(self, run_id: str) -> dict:
        return self._run_tb.get_entity(partition_key="run", row_key=run_id)

    def query_runs(self, query: str) -> list[BenchmarkRun]:
        entities = list(
            self._run_tb.query_entities(
                query, headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        runs = []
        for entity in entities:
            entity["steps"] = json.loads(entity["steps"])
            entity["changeStats"] = json.loads(entity["changeStats"])
            runs.append(fromdict(BenchmarkRun, entity))
        return runs

    def find_all_runs(self) -> list[BenchmarkRun]:
        entities = list(
            self._run_tb.list_entities(
                headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        runs = []
        for entity in entities:
            entity["steps"] = json.loads(entity["steps"])
            entity["changeStats"] = json.loads(entity["changeStats"])
            runs.append(fromdict(BenchmarkRun, entity))
        return runs

    def insert_tuning_run(self, run: TuningRun) -> str:
        run_obj = flatten_entry(run, row_key=run._id, partition="tuningrun")
        self._tuning_run_tb.upsert_entity(run_obj)
        return run._id

    def update_tuning_run(self, run_id: str, update_dict: dict[str, Any]):
        entity = self._tuning_run_tb.get_entity(
            partition_key="tuningrun", row_key=run_id
        )
        entity.update(update_dict)
        entity = flatten_entry(entity)
        self._tuning_run_tb.update_entity(entity)

    def delete_tuning_run(self, run_id: str):
        self._tuning_run_tb.delete_entity(partition_key="tuningrun", row_key=run_id)

    def delete_tuning_runs(self, query: str):
        queried_runs = self._tuning_run_tb.query_entities(query)
        for run in queried_runs:
            self._tuning_run_tb.delete_entity(run["PartitionKey"], run["RowKey"])

    def query_tuning_runs(self, query: str) -> list[TuningRun]:
        entities = list(
            self._tuning_run_tb.query_entities(
                query, headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        runs = []
        for entity in entities:
            entity["changeStats"] = json.loads(entity["changeStats"])
            runs.append(fromdict(TuningRun, entity))
        return runs

    def find_all_tuning_runs(self) -> list[TuningRun]:
        entities = list(
            self._tuning_run_tb.list_entities(
                headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        runs = []
        for entity in entities:
            entity["changeStats"] = json.loads(entity["changeStats"])
            runs.append(fromdict(TuningRun, entity))
        return runs

    def insert_pull_request(self, pr: RepoPullRequest) -> str:
        entity = flatten_entry(pr, row_key=pr._id, partition="pull")
        self._repo_tb.upsert_entity(entity)
        return pr._id

    def update_pull_request(self, prId: str, update_dict: dict[str, Any]):
        entity = self._repo_tb.get_entity(partition_key="pull", row_key=prId)
        entity.update(flatten_entry(update_dict))
        self._repo_tb.update_entity(entity)

    def delete_pull_request(self, prId: str) -> str:
        self._repo_tb.delete_entity(partition_key="pull", row_key=prId)
        return prId

    def insert_merge(self, pr: RepoMerge) -> str:
        entity = flatten_entry(pr, row_key=pr._id, partition="merge")
        self._repo_tb.upsert_entity(entity)
        return pr._id

    def update_merge(self, mergeId: str, update_dict: dict[str, Any]):
        entity = self._repo_tb.get_entity(partition_key="merge", row_key=mergeId)
        entity.update(flatten_entry(update_dict))
        self._repo_tb.update_entity(entity)

    def delete_merge(self, mergeId: str) -> str:
        self._repo_tb.delete_entity(partition_key="merge", row_key=mergeId)
        return mergeId

    def query_modifications(self, query: str) -> list[dict]:
        return list(self._repo_tb.query_entities(query))

    def find_modification_by_id(
        self, type: str, mod_id: str
    ) -> RepoPullRequest | RepoMerge:
        entity = self._repo_tb.get_entity(partition_key=type, row_key=mod_id)
        return (
            fromdict(RepoMerge, entity)
            if type == "merge"
            else fromdict(RepoPullRequest, entity)
        )

    def find_all_modifications(self) -> list[RepoModification]:
        entities = list(self._repo_tb.list_entities())
        entities_sorted = sorted(entities, key=lambda x: x["timestamp"], reverse=True)

        modifications = []
        for entity in entities_sorted:
            entity_dict = entity.copy()
            entity_dict["author"] = json.loads(entity_dict["author"])
            if entity_dict["type"] == "pr":
                entity_dict["commits"] = json.loads(entity_dict["commits"])
                modifications.append(fromdict(RepoPullRequest, entity_dict))
            else:
                modifications.append(fromdict(RepoMerge, entity_dict))

        return modifications

    def find_latest_pr(self) -> RepoPullRequest:
        entities = list(
            self._repo_tb.query_entities(
                "type eq 'pr'", headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        latest_entity = max(entities, key=lambda x: x["timestamp"])
        latest_entity["author"] = json.loads(latest_entity["author"])
        latest_entity["commits"] = json.loads(latest_entity["commits"])
        return fromdict(RepoPullRequest, latest_entity)

    def clear_all_runs(self):
        entities = self._run_tb.list_entities()
        for entity in entities:
            self._run_tb.delete_entity(
                partition_key=entity["PartitionKey"], row_key=entity["RowKey"]
            )

    def clear_all_repos(self):
        entities = self._repo_tb.list_entities()
        for entity in entities:
            self._repo_tb.delete_entity(
                partition_key=entity["PartitionKey"], row_key=entity["RowKey"]
            )

    # PerformanceRun methods
    def insert_performance(self, performance: PerformanceRun) -> str:
        performance_obj = flatten_entry(
            performance, row_key=performance._id, partition="performance"
        )
        self._performance_tb.upsert_entity(performance_obj)
        return performance._id

    def update_performance(self, performance_id: str, update_dict: dict[str, Any]):
        entity = self._performance_tb.get_entity(
            partition_key="performance", row_key=performance_id
        )
        entity.update(flatten_entry(update_dict))
        self._performance_tb.update_entity(entity)

    def delete_performance(self, performance_id: str):
        self._performance_tb.delete_entity(
            partition_key="performance", row_key=performance_id
        )

    def delete_performances(self, query: str):
        queried_performances = self._performance_tb.query_entities(query)
        for performance in queried_performances:
            self._performance_tb.delete_entity(
                performance["PartitionKey"], performance["RowKey"]
            )

    def clear_all_performances(self):
        entities = self._performance_tb.list_entities()
        for entity in entities:
            self._performance_tb.delete_entity(
                partition_key=entity["PartitionKey"], row_key=entity["RowKey"]
            )

    # Kernel methods
    def insert_kernels(self, kernels: list[Kernel]):
        kernel_entities = [
            flatten_entry(kernel, row_key=kernel.id, partition="kernel")
            for kernel in kernels
        ]
        self._batch_submit(self._kernel_tb, "upsert", kernel_entities)

    def update_kernels(self, kernels: list[Kernel]):
        for kernel in kernels:
            entity = self._kernel_tb.get_entity(
                partition_key="kernel", row_key=kernel.id
            )
            kernel_dict = asdict(kernel)
            entity.update(flatten_entry(kernel_dict))
            self._kernel_tb.update_entity(entity)

    def delete_kernels(self, kernel_ids: list[int]):
        for kernel_id in kernel_ids:
            self._kernel_tb.delete_entity(
                partition_key="kernel", row_key=str(kernel_id)
            )

    def clear_all_kernels(self):
        entities = self._kernel_tb.list_entities()
        for entity in entities:
            self._kernel_tb.delete_entity(
                partition_key=entity["PartitionKey"], row_key=entity["RowKey"]
            )

    def insert_tuning_configs(self, configs: list[TuningConfig]):
        tuning_entities = [
            flatten_entry(config, row_key=config._id, partition="tuning")
            for config in configs
        ]
        self._batch_submit(self._tuning_tb, "upsert", tuning_entities)

    def insert_tuning_config(self, config: TuningConfig) -> str:
        tuning_obj = flatten_entry(config, row_key=config._id, partition="tuning")
        self._tuning_tb.upsert_entity(tuning_obj)
        return config._id

    def update_tuning_config(self, tuning_id: str, update_dict: dict[str, Any]):
        entity = self._tuning_tb.get_entity(partition_key="tuning", row_key=tuning_id)
        entity.update(flatten_entry(update_dict))
        self._tuning_tb.update_entity(entity)

    def delete_tuning_configs(self, tuning_ids: list[str]):
        for tuning_id in tuning_ids:
            self._tuning_tb.delete_entity(partition_key="tuning", row_key=tuning_id)

    def clear_all_tuning_configs(self):
        entities = self._tuning_tb.list_entities()
        for entity in entities:
            self._tuning_tb.delete_entity(
                partition_key=entity["PartitionKey"], row_key=entity["RowKey"]
            )

    def query_tuning_configs(self, query: str) -> list[TuningConfig]:
        entities = list(
            self._tuning_tb.query_entities(
                query, headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        tuning_configs = []
        for entity in entities:
            entity["result"] = json.loads(entity["result"])
            tuning_configs.append(fromdict(TuningConfig, entity))
        return tuning_configs

    def find_all_tuning_configs(self) -> dict[str, list[dict]]:
        entities = list(
            self._tuning_tb.list_entities(
                headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        tuning_configs: list[TuningConfig] = []
        for entity in entities:
            entity["result"] = json.loads(entity["result"])
            tuning_configs.append(fromdict(TuningConfig, entity))

        configs_by_kernel: dict[str, list[dict]] = {}

        for config in tuning_configs:
            configs_by_kernel.setdefault(config.kernel_name, []).append(asdict(config))

        for kernel_name in configs_by_kernel.keys():
            configs_by_kernel[kernel_name].sort(
                key=lambda x: x["timestamp"], reverse=True
            )

        return configs_by_kernel

    def find_latest_tuning_configs(self) -> dict[str, dict]:
        configs_by_kernel = self.find_all_tuning_configs()
        latest_config_by_kernel = {
            kernel_name: results[0]
            for kernel_name, results in configs_by_kernel.items()
            if len(results) > 0
        }
        return latest_config_by_kernel

    # Additional helper methods for querying
    def query_performances(self, query: str) -> list[PerformanceRun]:
        entities = list(
            self._performance_tb.query_entities(
                query, headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        performances = []
        for entity in entities:
            entity["changeStats"] = json.loads(entity["changeStats"])
            performances.append(fromdict(PerformanceRun, entity))
        return performances

    def find_performance_by_id(self, performance_id: str) -> PerformanceRun:
        entity = self._performance_tb.get_entity(
            partition_key="performance", row_key=performance_id
        )
        entity["changeStats"] = json.loads(entity["changeStats"])
        return fromdict(PerformanceRun, entity)

    def query_kernels(self, query: str) -> list[Kernel]:
        entities = list(
            self._kernel_tb.query_entities(
                query, headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        kernels = []
        for entity in entities:
            entity["allowedBackends"] = json.loads(entity["allowedBackends"])
            entity["problem"] = json.loads(entity["problem"])
            kernels.append(fromdict(Kernel, entity))
        return kernels

    def find_kernel_by_id(self, kernel_id: str) -> Kernel:
        entity = self._kernel_tb.get_entity(partition_key="kernel", row_key=kernel_id)
        entity["allowedBackends"] = json.loads(entity["allowedBackends"])
        entity["problem"] = json.loads(entity["problem"])
        return fromdict(Kernel, entity)

    def find_all_performances(self) -> list[PerformanceRun]:
        entities = list(
            self._performance_tb.list_entities(
                headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        performances = []
        for entity in entities:
            entity["changeStats"] = json.loads(entity["changeStats"])
            performances.append(fromdict(PerformanceRun, entity))

        # Sort by timestamp in descending order (most recent first)
        performances.sort(key=lambda x: x.timestamp, reverse=True)
        return performances

    def find_all_kernels(self) -> list[Kernel]:
        entities = list(
            self._kernel_tb.list_entities(
                headers={"Accept": "application/json;odata=nometadata"}
            )
        )
        kernels = []
        for entity in entities:
            entity["allowedBackends"] = json.loads(entity["allowedBackends"])
            entity["problem"] = json.loads(entity["problem"])
            kernels.append(fromdict(Kernel, entity))
        return kernels
