"""
Data type management for kernel benchmarks.

This module provides a clean abstraction for handling data types across
different backends (PyTorch, IREE) and machines (different GPU architectures).

Key concepts:
- Display names: Simple, user-facing names ("f8", "f16", "bf16")
- Machine-specific variants: Architecture-dependent specs (gfx950 vs gfx942)
- Backend representations: PyTorch dtypes, IREE strings, etc.
"""

from .dtype_spec import DTypeSpec
from .registry import DTypeRegistry
from .device_context import DeviceContext

__all__ = [
    "DTypeSpec",
    "DTypeRegistry",
    "DeviceContext",
    "dtype_to_bytes",
]


# Simple display name -> byte count mapping for configs
# This is machine-independent and used for arithmetic intensity calculations
_DTYPE_BYTES = {
    "f8": 1,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
}


def dtype_to_bytes(display_name: str) -> int:
    """
    Get byte count for a dtype display name.
    """
    if display_name not in _DTYPE_BYTES:
        raise ValueError(
            f"Unknown dtype '{display_name}'. "
            f"Supported: {list(_DTYPE_BYTES.keys())}"
        )
    return _DTYPE_BYTES[display_name]
