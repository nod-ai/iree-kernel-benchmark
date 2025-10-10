from typing import Optional
import torch

from .dtype_spec import DTypeSpec


class DTypeRegistry:
    """
    Registry mapping display names to machine-specific DTypeSpecs.

    Handles machine-specific type resolution. For example, "f8" resolves to
    different variants depending on the target architecture:
    - gfx950: f8E4M3FN
    - Other: f8E4M3FNUZ

    Attributes:
        target: HIP target string (e.g., "gfx942", "gfx950")

    Examples:
        >>> registry = DTypeRegistry("gfx950")
        >>> spec = registry.resolve("f8")
        >>> spec.canonical_name
        'f8E4M3FN'

        >>> registry = DTypeRegistry("gfx942")
        >>> spec = registry.resolve("f8")
        >>> spec.canonical_name
        'f8E4M3FNUZ'
    """

    def __init__(self, target: str):
        """
        Initialize registry for a specific target architecture.

        Args:
            target: HIP target string (e.g., "gfx942", "gfx950")
        """
        self.target = target
        self._registry: dict[str, DTypeSpec] = {}
        self._torch_to_spec: dict[torch.dtype, DTypeSpec] = {}
        self._iree_to_spec: dict[str, DTypeSpec] = {}
        self._build_registry()

    def _build_registry(self):
        """Build the registry with all supported types for this target."""

        # Floating point types - standard across all architectures
        self._register(
            DTypeSpec(
                display_name="f16",
                canonical_name="float16",
                bits=16,
                is_float=True,
                torch_dtype=torch.float16,
                iree_string="f16",
            )
        )

        self._register(
            DTypeSpec(
                display_name="bf16",
                canonical_name="bfloat16",
                bits=16,
                is_float=True,
                torch_dtype=torch.bfloat16,
                iree_string="bf16",
            )
        )

        self._register(
            DTypeSpec(
                display_name="f32",
                canonical_name="float32",
                bits=32,
                is_float=True,
                torch_dtype=torch.float32,
                iree_string="f32",
            )
        )

        self._register(
            DTypeSpec(
                display_name="f64",
                canonical_name="float64",
                bits=64,
                is_float=True,
                torch_dtype=torch.float64,
                iree_string="f64",
            )
        )

        # 8-bit float types - architecture dependent
        if self.target == "gfx950":
            # gfx950 (MI350X, MI355X) uses FN variant
            self._register(
                DTypeSpec(
                    display_name="f8",
                    canonical_name="f8E4M3FN",
                    bits=8,
                    is_float=True,
                    torch_dtype=torch.float8_e4m3fn,
                    iree_string="f8E4M3FN",
                )
            )
        else:
            # Other architectures use FNUZ variant
            self._register(
                DTypeSpec(
                    display_name="f8",
                    canonical_name="f8E4M3FNUZ",
                    bits=8,
                    is_float=True,
                    torch_dtype=torch.float8_e4m3fnuz,
                    iree_string="f8E4M3FNUZ",
                )
            )

        # Integer types
        self._register(
            DTypeSpec(
                display_name="i8",
                canonical_name="int8",
                bits=8,
                is_float=False,
                torch_dtype=torch.int8,
                iree_string="i8",
            )
        )

        self._register(
            DTypeSpec(
                display_name="i16",
                canonical_name="int16",
                bits=16,
                is_float=False,
                torch_dtype=torch.int16,
                iree_string="i16",
            )
        )

        self._register(
            DTypeSpec(
                display_name="i32",
                canonical_name="int32",
                bits=32,
                is_float=False,
                torch_dtype=torch.int32,
                iree_string="i32",
            )
        )

        self._register(
            DTypeSpec(
                display_name="i64",
                canonical_name="int64",
                bits=64,
                is_float=False,
                torch_dtype=torch.int64,
                iree_string="i64",
            )
        )

        # Boolean type
        self._register(
            DTypeSpec(
                display_name="bool",
                canonical_name="bool",
                bits=1,
                is_float=False,
                torch_dtype=torch.bool,
                iree_string="i1",  # IREE uses i1 for bool
            )
        )

    def _register(self, spec: DTypeSpec):
        """
        Register a dtype spec in all lookup tables.

        Args:
            spec: DTypeSpec to register
        """
        # Register by display name (primary lookup)
        self._registry[spec.display_name.lower()] = spec

        # Register by canonical name (for reverse lookups)
        self._registry[spec.canonical_name.lower()] = spec

        # Register by torch dtype
        self._torch_to_spec[spec.torch_dtype] = spec

        # Register by IREE string
        self._iree_to_spec[spec.iree_string.lower()] = spec

    def resolve(self, display_name: str) -> DTypeSpec:
        """
        Resolve display name to machine-specific DTypeSpec.

        Args:
            display_name: Display name (e.g., "f8", "f16", "bf16")

        Returns:
            Machine-specific DTypeSpec

        Raises:
            KeyError: If display name is not recognized

        Examples:
            >>> registry = DTypeRegistry("gfx950")
            >>> spec = registry.resolve("f8")
            >>> spec.canonical_name
            'f8E4M3FN'
        """
        normalized = display_name.lower().strip().replace("_", "")
        spec = self._registry.get(normalized)
        if spec is None:
            raise KeyError(
                f"Unknown dtype '{display_name}' for target {self.target}. "
                f"Available types: {list(set(s.display_name for s in self._registry.values()))}"
            )
        return spec

    def from_torch(self, torch_dtype: torch.dtype) -> DTypeSpec:
        """
        Convert PyTorch dtype to DTypeSpec.

        Args:
            torch_dtype: PyTorch dtype object

        Returns:
            Corresponding DTypeSpec

        Raises:
            KeyError: If torch dtype is not recognized
        """
        spec = self._torch_to_spec.get(torch_dtype)
        if spec is None:
            raise KeyError(f"Unknown PyTorch dtype: {torch_dtype}")
        return spec

    def from_iree_string(self, iree_str: str) -> DTypeSpec:
        """
        Convert IREE string to DTypeSpec.

        Args:
            iree_str: IREE string representation (e.g., "f16", "f8E4M3FNUZ")

        Returns:
            Corresponding DTypeSpec

        Raises:
            KeyError: If IREE string is not recognized
        """
        normalized = iree_str.lower().strip()
        spec = self._iree_to_spec.get(normalized)
        if spec is None:
            raise KeyError(f"Unknown IREE dtype string: {iree_str}")
        return spec

    def list_types(self) -> list[str]:
        """
        Get list of all available display names.

        Returns:
            List of display names sorted alphabetically
        """
        # Get unique display names (avoid duplicates from canonical lookups)
        display_names = set(spec.display_name for spec in self._registry.values())
        return sorted(display_names)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"DTypeRegistry(target='{self.target}', types={self.list_types()})"
