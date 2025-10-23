"""
Kernel configuration management module.

This module provides centralized configuration handling for kernel benchmarks,
including type definitions, registry, and loading utilities.

Main components:
- OpConfig: Abstract base class for all configurations
- ConfigRegistry: Centralized registration system
- Config loaders: JSON/CSV loading utilities
- Type-specific configs: GemmConfig, ConvConfig, etc.
"""

from kernel_bench.config.base import OpConfig
from kernel_bench.config.registry import (
    ConfigRegistry,
    get_global_registry,
    register_config,
)
from kernel_bench.config.loaders import (
    load_configs,
    load_configs_from_json,
    save_configs_to_json,
)
