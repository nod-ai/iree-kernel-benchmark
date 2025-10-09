"""
Configuration registry for kernel benchmarks.

This module provides a centralized registry for kernel configuration types,
replacing the scattered CONFIG_CLASSES dictionaries.
"""

from typing import Type, Dict, Optional, Callable
from kernel_bench.config.base import OpConfig


class ConfigRegistry:
    """
    Centralized registry for kernel configuration types and their loaders.

    This class manages the mapping between kernel types and their associated
    configuration classes and default problem loaders.
    """

    def __init__(self):
        self._config_classes: Dict[str, Type[OpConfig]] = {}
        self._config_loaders: Dict[str, Callable] = {}

    def register(
        self,
        kernel_type: str,
        config_class: Type[OpConfig],
        loader_func: Optional[Callable] = None,
    ) -> None:
        """
        Register a configuration type for a kernel.

        Args:
            kernel_type: The kernel type identifier (e.g., "gemm", "attention")
            config_class: The configuration class (must inherit from OpConfig)
            loader_func: Optional function to load default problem sets
        """
        if not issubclass(config_class, OpConfig):
            raise TypeError(
                f"Config class {config_class.__name__} must inherit from OpConfig"
            )

        self._config_classes[kernel_type] = config_class

        if loader_func is not None:
            self._config_loaders[kernel_type] = loader_func

    def get_config_class(self, kernel_type: str) -> Type[OpConfig]:
        """
        Get the configuration class for a kernel type.

        Args:
            kernel_type: The kernel type identifier

        Returns:
            The configuration class for the kernel type

        Raises:
            KeyError: If kernel type is not registered
        """
        if kernel_type not in self._config_classes:
            raise KeyError(
                f"No config class registered for kernel type '{kernel_type}'. "
                f"Available types: {list(self._config_classes.keys())}"
            )
        return self._config_classes[kernel_type]

    def get_loader(self, kernel_type: str) -> Optional[Callable]:
        """
        Get the default problem loader for a kernel type.

        Args:
            kernel_type: The kernel type identifier

        Returns:
            The loader function, or None if not registered
        """
        return self._config_loaders.get(kernel_type)

    def has_kernel_type(self, kernel_type: str) -> bool:
        """Check if a kernel type is registered."""
        return kernel_type in self._config_classes

    def list_kernel_types(self) -> list[str]:
        """Get list of all registered kernel types."""
        return list(self._config_classes.keys())

    def as_dict(self) -> Dict[str, Type[OpConfig]]:
        """
        Get dictionary representation for backwards compatibility.

        Returns:
            Dictionary mapping kernel types to config classes
        """
        return self._config_classes.copy()


# Global singleton instance
_global_registry = ConfigRegistry()


def get_global_registry() -> ConfigRegistry:
    """Get the global configuration registry instance."""
    return _global_registry


def register_config(
    kernel_type: str,
    config_class: Type[OpConfig],
    loader_func: Optional[Callable] = None,
) -> None:
    """
    Convenience function to register a config with the global registry.

    Args:
        kernel_type: The kernel type identifier
        config_class: The configuration class
        loader_func: Optional default problem loader
    """
    _global_registry.register(kernel_type, config_class, loader_func)
