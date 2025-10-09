"""
Configuration loaders for kernel benchmarks.

This module provides utilities for loading problem sets from various sources
(CSV files, JSON files, programmatic generators).
"""

import json
import csv
from pathlib import Path
from typing import List, Tuple, Type, Any, Optional
from dataclass_wizard import fromdict

from kernel_bench.config.base import OpConfig


def load_configs_from_json(
    json_path: Path,
    config_class: Type[OpConfig],
) -> List[Tuple[str, OpConfig]]:
    """
    Load configurations from a JSON file.

    Args:
        json_path: Path to JSON file
        config_class: Configuration class to deserialize into

    Returns:
        List of (tag, config) tuples
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    configs = []
    for item in data:
        if isinstance(item, dict):
            # Simple dict format
            tag = item.get("tag", "default")
            config = fromdict(config_class, item)
            configs.append((tag, config))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Tuple format: (tag, config_dict)
            tag, config_dict = item
            config = fromdict(config_class, config_dict)
            configs.append((tag, config))
        else:
            raise ValueError(f"Invalid config format in JSON: {item}")

    return configs


def load_configs_from_csv(
    csv_path: Path,
    config_class: Type[OpConfig],
) -> List[Tuple[str, OpConfig]]:
    """
    Load configurations from a CSV file.

    Args:
        csv_path: Path to CSV file
        config_class: Configuration class to deserialize into

    Returns:
        List of (tag, config) tuples
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        configs = []

        for row in reader:
            # Remove empty strings and convert types
            cleaned_row = {
                k: v
                for k, v in row.items()
                if v and k in config_class.__dataclass_fields__
            }

            tag = row.get("tag", "default")
            config = fromdict(config_class, cleaned_row)
            configs.append((tag, config))

    return configs


def load_configs(
    file_path: Path,
    config_class: Type[OpConfig],
) -> List[Tuple[str, OpConfig]]:
    """
    Load configurations from a file (auto-detects format).

    Args:
        file_path: Path to configuration file
        config_class: Configuration class to deserialize into

    Returns:
        List of (tag, config) tuples
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    if file_path.suffix == ".json":
        return load_configs_from_json(file_path, config_class)
    elif file_path.suffix == ".csv":
        return load_configs_from_csv(file_path, config_class)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_configs_to_json(
    configs: List[Tuple[str, OpConfig]],
    json_path: Path,
) -> None:
    """
    Save configurations to a JSON file.

    Args:
        configs: List of (tag, config) tuples
        json_path: Output path for JSON file
    """
    data = []
    for tag, config in configs:
        config_dict = config.to_dict()
        config_dict["tag"] = tag
        data.append(config_dict)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def save_configs_to_csv(
    configs: List[Tuple[str, OpConfig]],
    csv_path: Path,
) -> None:
    """
    Save configurations to a CSV file.

    Args:
        configs: List of (tag, config) tuples
        csv_path: Output path for CSV file
    """
    if not configs:
        return

    # Get fieldnames from first config
    fieldnames = ["tag"] + list(configs[0][1].to_dict().keys())

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tag, config in configs:
            row = {"tag": tag, **config.to_dict()}
            writer.writerow(row)
