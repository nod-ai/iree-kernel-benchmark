from dataclasses import dataclass, field
from typing import Optional
import torch

from .dtype_spec import DTypeSpec
from .registry import DTypeRegistry


HIP_TARGETS = {
    "mi100": "gfx908",
    "mi210": "gfx90a",
    "mi250": "gfx90a",
    "mi300a": "gfx942",
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi350x": "gfx950",
    "mi355x": "gfx950",
    "v710": "gfx1101",
    "w7700": "gfx1101",
    "w7800": "gfx1100",
    "w7900": "gfx1100",
    "rx7700xt": "gfx1101",
    "rx7800xt": "gfx1101",
    "rx7900xt": "gfx1100",
    "rx7900xtx": "gfx1100",
    "rx9060xt": "gfx1200",
    "rx9070": "gfx1201",
    "rx9070xt": "gfx1201",
    "r9700": "gfx1201",
}


def machine_to_hip_target(machine_name: str) -> str:
    """
    Convert machine name to HIP target.

    Args:
        machine_name: Machine name (e.g., "mi300x", "MI325X")

    Returns:
        HIP target string (e.g., "gfx942", "gfx950")

    Raises:
        ValueError: If machine name is not recognized
    """
    target = HIP_TARGETS.get(machine_name.lower().strip())
    if target is None:
        raise ValueError(
            f"Unknown machine '{machine_name}'. "
            f"Supported machines: {list(HIP_TARGETS.keys())}"
        )
    return target


def get_hip_target_from_device() -> str:
    """
    Detect HIP target from current device.

    Returns:
        HIP target string (e.g., "gfx942")

    Raises:
        RuntimeError: If ROCm not available or no GPUs detected
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Cannot detect HIP target: PyTorch not compiled with ROCm.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Cannot detect HIP target: No AMD GPUs detected.")

    arch_name = torch.cuda.get_device_properties().gcnArchName
    target = arch_name.split(":")[0]
    return target


def get_shared_memory_limit(hip_target: Optional[str] = None) -> float:
    if not hip_target:
        hip_target = get_hip_target_from_device()

    if hip_target.startswith("gfx95"):
        return 163840
    else:
        return 65536


@dataclass
class DeviceContext:
    """
    Complete device execution context.

    Manages machine info, device IDs, and provides type conversions
    for different backends.

    Attributes:
        machine: Machine name (e.g., "MI300X", "MI325X")
        hip_target: HIP target string (e.g., "gfx942", "gfx950")
        device_id: Optional device ID for multi-GPU systems
        _dtype_registry: Internal registry for type conversions

    Examples:
        >>> ctx = DeviceContext.from_machine("mi300x")
        >>> ctx.dtype_to_torch("f8")
        torch.float8_e4m3fnuz
        >>> ctx.dtype_to_iree("f8")
        'f8E4M3FNUZ'

        >>> ctx = DeviceContext.from_machine("mi350x")
        >>> ctx.dtype_to_torch("f8")
        torch.float8_e4m3fn
        >>> ctx.dtype_to_iree("f8")
        'f8E4M3FN'
    """

    machine: str
    hip_target: str
    device_id: Optional[int] = None
    _dtype_registry: DTypeRegistry = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the dtype registry for this target."""
        self._dtype_registry = DTypeRegistry(self.hip_target)
        # Normalize machine name to uppercase
        self.machine = self.machine.upper().strip()

    def resolve_dtype(self, display_name: str) -> DTypeSpec:
        """Get machine-specific dtype spec from display name."""
        return self._dtype_registry.resolve(display_name)

    def dtype_to_torch(self, display_name: str) -> torch.dtype:
        """Convert display name directly to PyTorch dtype."""
        return self.resolve_dtype(display_name).to_torch()

    def dtype_to_iree(self, display_name: str) -> str:
        """Convert display name directly to IREE string."""
        return self.resolve_dtype(display_name).to_iree_string()

    def dtype_from_torch(self, torch_dtype: torch.dtype) -> DTypeSpec:
        """Convert PyTorch dtype back to DTypeSpec."""
        return self._dtype_registry.from_torch(torch_dtype)

    def dtype_from_iree(self, iree_str: str) -> DTypeSpec:
        """Convert IREE string back to DTypeSpec."""
        return self._dtype_registry.from_iree_string(iree_str)

    def hip_device_str(self) -> str:
        """Get HIP device string for IREE."""
        if self.device_id is None:
            return "hip"
        return f"hip://{self.device_id}"

    def cuda_device_str(self) -> str:
        """Get CUDA device string for PyTorch."""
        if self.device_id is None:
            return "cuda"
        return f"cuda:{self.device_id}"

    def list_supported_types(self) -> list[str]:
        """Get list of all supported dtype display names."""
        return self._dtype_registry.list_types()

    @classmethod
    def from_machine(
        cls, machine: str, device_id: Optional[int] = None
    ) -> "DeviceContext":
        """Create DeviceContext from machine name."""
        hip_target = machine_to_hip_target(machine)
        return cls(machine=machine, hip_target=hip_target, device_id=device_id)

    @classmethod
    def from_current_device(cls, device_id: Optional[int] = None) -> "DeviceContext":
        """Create DeviceContext by auto-detecting current device."""
        hip_target = get_hip_target_from_device()

        # Try to find matching machine name (for display purposes)
        machine = None
        for machine_name, target in HIP_TARGETS.items():
            if target == hip_target:
                machine = machine_name
                break

        if machine is None:
            machine = hip_target  # Fall back to target as machine name

        return cls(machine=machine, hip_target=hip_target, device_id=device_id)

    def __repr__(self) -> str:
        """String representation for debugging."""
        device_str = f", device={self.device_id}" if self.device_id is not None else ""
        return f"DeviceContext(machine='{self.machine}', target='{self.hip_target}'{device_str})"
