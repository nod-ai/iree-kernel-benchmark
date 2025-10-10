from dataclasses import dataclass, field
from typing import Any, Optional
import torch


@dataclass(frozen=True)
class DTypeSpec:
    """
    Immutable specification for a data type.

    This class represents a complete specification of a data type, including
    its display name (for UI/serialization), canonical name (machine-specific),
    and metadata needed for conversions.

    Attributes:
        display_name: Simple name for UI/serialization ("f8", "f16", "bf16")
        canonical_name: Full specification ("f8E4M3FNUZ", "float16", etc.)
        bits: Bit width (8, 16, 32, 64)
        is_float: Whether this is a floating point type
        torch_dtype: PyTorch dtype representation (cached)
        iree_string: IREE string representation (cached)
        backend_hints: Optional backend-specific metadata

    Examples:
        >>> spec = DTypeSpec(
        ...     display_name="f8",
        ...     canonical_name="f8E4M3FNUZ",
        ...     bits=8,
        ...     is_float=True,
        ...     torch_dtype=torch.float8_e4m3fnuz,
        ...     iree_string="f8E4M3FNUZ"
        ... )
        >>> spec.to_torch()
        torch.float8_e4m3fnuz
        >>> spec.to_iree_string()
        'f8E4M3FNUZ'
    """

    display_name: str
    canonical_name: str
    bits: int
    is_float: bool
    torch_dtype: torch.dtype
    iree_string: str
    backend_hints: dict[str, Any] = field(default_factory=dict)

    def to_torch(self) -> torch.dtype:
        """
        Convert to PyTorch dtype.

        Returns:
            PyTorch dtype object
        """
        return self.torch_dtype

    def to_iree_string(self) -> str:
        """
        Convert to IREE string representation.

        Returns:
            IREE-compatible string (e.g., "f16", "f8E4M3FNUZ")
        """
        return self.iree_string

    def bitwidth(self) -> int:
        """
        Get number of bits for this dtype.

        Returns:
            Number of bits (minimum 1)
        """
        return self.bits

    def num_bytes(self) -> int:
        """
        Get number of bytes for this dtype.

        Returns:
            Number of bytes (minimum 1)
        """
        return max(1, self.bits // 8)

    def max_value(self) -> float | int:
        """
        Get maximum representable value for this dtype.

        Returns:
            Maximum value (float for floating point, int for integer types)
        """
        if self.is_float:
            return torch.finfo(self.torch_dtype).max
        else:
            return torch.iinfo(self.torch_dtype).max

    def __str__(self) -> str:
        """String representation uses display name."""
        return self.display_name

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"DTypeSpec(display='{self.display_name}', "
            f"canonical='{self.canonical_name}', "
            f"bits={self.bits})"
        )
