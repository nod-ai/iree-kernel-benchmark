"""
Base configuration classes for kernel benchmarking.

This module defines the abstract base class for all kernel configurations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclass_wizard import asdict
from typing import Any, Dict, List


@dataclass
class OpConfig(ABC):
    """
    Abstract base class for kernel operation configurations.

    All kernel configurations (GEMM, Attention, Conv, etc.) must inherit from
    this class and implement the required methods for benchmarking metadata.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get a unique identifier for this configuration.

        Returns:
            A string that uniquely identifies this configuration, typically
            encoding the key dimensions and parameters.
        """
        pass

    @abstractmethod
    def get_flops(self) -> int:
        """
        Calculate the number of floating point operations for this configuration.

        Returns:
            Total FLOPS count as an integer.
        """
        pass

    @abstractmethod
    def get_byte_count(self) -> int:
        """
        Calculate the number of bytes transferred for this configuration.

        Returns:
            Total byte count as an integer (for arithmetic intensity calculation).
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    def get_dim_names(self) -> List[str]:
        """
        Get the names of all configuration dimensions.

        Returns:
            List of dimension names (field names from the dataclass).
        """
        return list(self.to_dict().keys())
