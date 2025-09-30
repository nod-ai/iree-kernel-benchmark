"""
Repository pattern implementation for Azure Table Storage with full static type safety.
Provides a clean, type-safe interface for database operations without dynamic class creation.
"""

import os
import json
from dataclasses import is_dataclass, asdict, fields
from dataclass_wizard import fromdict
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
)
from azure.data.tables import TableServiceClient, TableClient
from azure.data.tables._deserialize import TablesEntityDatetime
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


class DatabaseSerializer:
    """Handles serialization and deserialization of complex data types for database storage."""

    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Serialize a value for database storage."""
        if value is None:
            return value
        elif isinstance(value, datetime):
            # Azure Table Storage handles datetime objects natively
            return value
        elif is_dataclass(value):
            # Convert dataclass to dict, then to JSON string
            try:
                return json.dumps(asdict(value), default=str)
            except Exception as e:
                print(f"Error serializing dataclass {type(value)}: {e}")
                return str(value)
        elif isinstance(value, (dict, list)):
            # Convert complex JSON structures to strings
            try:
                return json.dumps(value, default=str)
            except Exception as e:
                print(f"Error serializing dict/list {type(value)}: {e}")
                return str(value)
        else:
            # Return primitive types as-is
            return value

    @staticmethod
    def serialize_entity(
        obj: Any, row_key: str = None, partition: str = None
    ) -> Dict[str, Any]:
        """Serialize an entire entity for database storage."""
        if not is_dataclass(obj):
            raise ValueError("Object must be a dataclass")

        new_obj = {}
        if row_key:
            new_obj["RowKey"] = row_key
        if partition:
            new_obj["PartitionKey"] = partition

        obj_dict = asdict(obj)
        for key, value in obj_dict.items():
            new_obj[key] = DatabaseSerializer.serialize_value(value)

        return new_obj

    @staticmethod
    def deserialize_entity(entity_dict: Dict[str, Any], target_class: Type[T]) -> T:
        """Deserialize an entity from database storage back to dataclass."""
        try:
            # Clean up Azure metadata and handle special Azure types
            deserialized = {}
            for field_name, field_value in entity_dict.items():
                if field_name in ("PartitionKey", "RowKey", "etag", "Timestamp"):
                    continue  # Skip Azure table metadata

                # Handle Azure Table Storage special types
                if isinstance(field_value, TablesEntityDatetime):
                    # Convert TablesEntityDatetime to Python datetime
                    deserialized[field_name] = datetime(
                        year=field_value.year,
                        month=field_value.month,
                        day=field_value.day,
                        hour=field_value.hour,
                        minute=field_value.minute,
                        second=field_value.second,
                        microsecond=field_value.microsecond,
                        tzinfo=field_value.tzinfo,
                    )
                elif isinstance(field_value, str):
                    try:
                        # Try to parse as JSON
                        parsed = json.loads(field_value)
                        deserialized[field_name] = parsed
                    except (json.JSONDecodeError, TypeError):
                        # If JSON parsing fails, keep as string
                        deserialized[field_name] = field_value
                else:
                    deserialized[field_name] = field_value

            # Let dataclass_wizard handle the type conversion
            return fromdict(target_class, deserialized)

        except Exception as e:
            print(f"Error deserializing entity for {target_class.__name__}: {e}")
            # Better fallback: create a more robust fallback data structure
            try:
                fallback_data = {}
                for field_name, field_value in entity_dict.items():
                    if field_name not in (
                        "PartitionKey",
                        "RowKey",
                        "etag",
                        "Timestamp",
                    ):
                        fallback_data[field_name] = field_value

                return fromdict(target_class, fallback_data)
            except Exception as fallback_error:
                print(f"Fallback deserialization also failed: {fallback_error}")
                raise e


class DatabaseRepository(Generic[T]):
    """
    Generic repository for database operations with full static type safety.

    Usage:
        @dataclass
        class TuningConfig:
            _id: str
            name: str
            value: int

        # Create repository with full type safety
        tuning_configs = DatabaseRepository(TuningConfig, "tuningconfigs")

        # All operations are fully typed
        config = TuningConfig(_id="123", name="test", value=42)
        tuning_configs.upsert(config)  # ✅ Type-safe
        found = tuning_configs.find_by_id("123")  # ✅ Returns Optional[TuningConfig]
        all_configs = tuning_configs.find_all()  # ✅ Returns List[TuningConfig]
    """

    def __init__(self, model_class: Type[T], table: str):
        """
        Initialize repository for a specific model and table.

        Args:
            model_class: The dataclass type this repository manages
            table: The Azure Table Storage table name
        """
        if not is_dataclass(model_class):
            raise ValueError("Model class must be a dataclass")

        field_names = [f.name for f in fields(model_class)]
        if "_id" not in field_names:
            raise ValueError("Model class must have an '_id' field")

        self.model_class = model_class
        self.table = table
        self._service_client: Optional[TableServiceClient] = None
        self._table_client: Optional[TableClient] = None

    def _get_clients(self) -> tuple[TableServiceClient, TableClient]:
        """Get database clients - lazy initialization."""
        if self._service_client is None:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                raise ValueError(
                    "AZURE_STORAGE_CONNECTION_STRING environment variable is required"
                )

            self._service_client = TableServiceClient.from_connection_string(
                connection_string
            )
            self._table_client = self._service_client.create_table_if_not_exists(
                self.table
            )
        return self._service_client, self._table_client

    def find_by_id(self, id: str) -> Optional[T]:
        """Find an entity by its ID."""
        try:
            _, table_client = self._get_clients()
            entity = table_client.get_entity(partition_key=self.table, row_key=id)
            return DatabaseSerializer.deserialize_entity(entity, self.model_class)
        except Exception:
            return None

    def find_all(self, query: Optional[Dict[str, Any]] = None) -> List[T]:
        """Find all entities, optionally filtered by query."""
        _, table_client = self._get_clients()

        if query:
            # Convert query dict to OData filter string
            filter_parts = []
            for key, value in query.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} eq '{value}'")
                else:
                    filter_parts.append(f"{key} eq {value}")
            query_filter = " and ".join(filter_parts)
            entities = list(
                table_client.query_entities(
                    query_filter,
                    headers={"Accept": "application/json;odata=nometadata"},
                )
            )
        else:
            entities = list(
                table_client.list_entities(
                    headers={"Accept": "application/json;odata=nometadata"}
                )
            )

        results = []
        for entity in entities:
            try:
                results.append(
                    DatabaseSerializer.deserialize_entity(entity, self.model_class)
                )
            except Exception as e:
                print(f"Warning: Failed to deserialize entity: {e}")
                continue

        return results

    def query(self, query_filter: str) -> List[T]:
        """Query entities using raw Azure OData filter syntax."""
        _, table_client = self._get_clients()

        try:
            entities = list(
                table_client.query_entities(
                    query_filter,
                    headers={"Accept": "application/json;odata=nometadata"},
                )
            )

            results = []
            for entity in entities:
                try:
                    results.append(
                        DatabaseSerializer.deserialize_entity(entity, self.model_class)
                    )
                except Exception as e:
                    print(f"Warning: Failed to deserialize entity: {e}")
                    continue

            return results
        except Exception as e:
            print(f"Error executing query '{query_filter}': {e}")
            return []

    def upsert(self, obj: T) -> bool:
        """Insert or update an entity."""
        try:
            _, table_client = self._get_clients()
            if not hasattr(obj, "_id"):
                raise ValueError("Object must have an _id attribute")

            entity = DatabaseSerializer.serialize_entity(
                obj, row_key=obj._id, partition=self.table
            )
            table_client.upsert_entity(entity)
            return True
        except Exception as e:
            print(f"Error upserting entity: {e}")
            return False

    def upsert_many(self, objects: List[T]) -> bool:
        """Insert or update multiple entities."""
        try:
            _, table_client = self._get_clients()

            # Process in batches of 100 (Azure Table Storage limit)
            # All entities in a batch must have the same partition key
            batch_size = 100
            for i in range(0, len(objects), batch_size):
                batch = objects[i : i + batch_size]

                for obj in batch:
                    if not hasattr(obj, "_id"):
                        raise ValueError("All objects must have an _id attribute")

                    entity = DatabaseSerializer.serialize_entity(
                        obj, row_key=obj._id, partition=self.table
                    )
                    table_client.upsert_entity(entity)

            return True
        except Exception as e:
            print(f"Error in upsert_many: {e}")
            return False

    def update_by_id(self, id: str, update: Dict[str, Any]) -> Optional[T]:
        """Update an entity by ID and return the updated object."""
        try:
            _, table_client = self._get_clients()
            entity = table_client.get_entity(partition_key=self.table, row_key=id)

            # Serialize the update values
            serialized_update = {}
            for key, value in update.items():
                serialized_update[key] = DatabaseSerializer.serialize_value(value)

            entity.update(serialized_update)
            table_client.update_entity(entity)

            # Return the updated object
            return self.find_by_id(id)
        except Exception as e:
            print(f"Error updating entity by ID: {e}")
            return None

    def delete_by_id(self, id: str) -> bool:
        """Delete an entity by ID."""
        try:
            _, table_client = self._get_clients()
            table_client.delete_entity(partition_key=self.table, row_key=id)
            return True
        except Exception as e:
            print(f"Error deleting entity by ID: {e}")
            return False

    def delete(self, obj: T) -> bool:
        """Delete an entity by object."""
        if not hasattr(obj, "_id"):
            return False
        return self.delete_by_id(obj._id)

    def count(self) -> int:
        """Count total entities in the table."""
        try:
            entities = self.find_all()
            return len(entities)
        except Exception as e:
            print(f"Error counting entities: {e}")
            return 0

    def exists(self, id: str) -> bool:
        """Check if an entity exists by ID."""
        return self.find_by_id(id) is not None

    def clear_all(self) -> bool:
        """Clear all entities from the table by deleting and recreating it."""
        try:
            service_client, _ = self._get_clients()
            service_client.delete_table(self.table)
            self._table_client = service_client.create_table_if_not_exists(self.table)
            return True
        except Exception as e:
            print(f"Error clearing all entities: {e}")
            return False


# Convenience function to create repositories
def create_repository(model_class: Type[T], table: str) -> DatabaseRepository[T]:
    """
    Convenience function to create a repository with proper typing.

    Usage:
        tuning_configs = create_repository(TuningConfig, "tuningconfigs")
    """
    return DatabaseRepository(model_class, table)
