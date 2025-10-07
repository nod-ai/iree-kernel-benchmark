from dataclasses import dataclass
from typing import Any, Iterable, Optional
import torch

from kernel_bench.utils.print_utils import get_logger

DTYPE_TO_TORCH = {
    "bf16": torch.bfloat16,
    "f8E5M2": torch.float8_e5m2,
    "f8E5M2FNUZ": torch.float8_e5m2fnuz,
    "f8E4M3FN": torch.float8_e4m3fn,
    "f8E4M3FNUZ": torch.float8_e4m3fnuz,
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}

DTYPE_TO_BITS = {
    "bf16": 16,
    "f8E5M2": 8,
    "f8E5M2FNUZ": 8,
    "f8E4M3FN": 8,
    "f8E4M3FNUZ": 8,
    "f16": 16,
    "f32": 32,
    "f64": 64,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "bool": 1,
}


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


class BenchDtype:
    def __init__(
        self,
        dtype_str: str,
        hip_target: Optional[str] = None,
    ):
        self.dtype_str = "f8" if "f8" in dtype_str else dtype_str
        self.hip_target = hip_target or get_hip_target()

    def to_torch(self) -> torch.dtype:
        return dtype_to_torch(self.dtype_str, self.hip_target)

    def to_string(self) -> str:
        return self.dtype_str

    def to_full_string(self) -> str:
        return get_device_specific_dtype(self.dtype_str, self.hip_target)

    def num_bytes(self) -> int:
        return dtype_to_bytes(self.to_torch())

    def num_bits(self) -> int:
        return dtype_to_bits(self.to_torch())

    def max_value(self) -> Any:
        return dtype_max_value(self.to_torch())

    def __str__(self):
        return self.dtype_str


class BenchDeviceContext:
    def __init__(self, machine: str, device_id: Optional[int] = None):
        self.machine = machine.upper().strip()
        self.target = machine_to_hip_target(machine)
        self.device_id = device_id

    def get_bench_dtype(self, dtype_str: str):
        return BenchDtype(dtype_str, self.target)

    def hip_device_str(self) -> str:
        if not self.device_id:
            return "hip"
        return f"hip://{self.device_id}"

    def cuda_device_str(self) -> str:
        if not self.device_id:
            return "cuda"
        return f"cuda://{self.device_id}"


def machine_to_hip_target(machine_name: str):
    target = HIP_TARGETS.get(machine_name.lower().strip())
    if target is None:
        get_logger().error(
            f"Could not find valid hip target for machine {machine_name}"
        )
    return target


def get_hip_target() -> str:
    if not torch.cuda.is_available():
        raise ValueError("Cannot get dtype: torch not compiled with ROCM.")
    if torch.cuda.device_count() < 1:
        raise ValueError("Cannot get dtype: torch cannot detect AMD GPUs.")
    arch_name = torch.cuda.get_device_properties().gcnArchName
    target = arch_name.split(":")[0]
    return target


def get_device_specific_dtype(dtype: str, target: Optional[str] = None) -> str:
    dtype = dtype.strip().replace("_", "")

    if dtype == "f8":
        if not target:
            target = get_hip_target()
        if target == "gfx950":
            dtype = "f8E4M3FN"
        else:
            dtype = "f8E4M3FNUZ"

    return dtype


def dtype_to_torch(dtype: str | torch.dtype, target: Optional[str] = None):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = get_device_specific_dtype(dtype, target)
    return DTYPE_TO_TORCH[dtype]


def dtype_to_bits(dtype: str | torch.dtype, target: Optional[str] = None):
    if isinstance(dtype, torch.dtype):
        dtype = torch_dtype_to_str(dtype)
    else:
        dtype = get_device_specific_dtype(dtype, target)
    return DTYPE_TO_BITS[dtype]


def dtype_to_bytes(dtype: str | torch.dtype, target: Optional[str] = None):
    return max(1, dtype_to_bits(dtype, target) // 8)


def dtype_max_value(dtype: str | torch.dtype, target: Optional[str] = None):
    if not isinstance(dtype, torch.dtype):
        dtype = dtype_to_torch(dtype, target)
    if dtype.is_floating_point:
        return torch.finfo(dtype).max
    else:
        return torch.iinfo(dtype).max


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    torch_dtype_to_str = {
        torch_dtype: dtype_str for dtype_str, torch_dtype in DTYPE_TO_TORCH.items()
    }
    dtype = torch_dtype_to_str.get(dtype)
    if not dtype:
        raise ValueError(f"Datatype {dtype} is invalid.")
    return dtype


def stringify_shape(shape: tuple[int, ...] | Any, dtype: str | torch.dtype) -> str:
    if isinstance(dtype, torch.dtype):
        dtype = torch_dtype_to_str(dtype)
    else:
        dtype = get_device_specific_dtype(dtype)
    if isinstance(shape, tuple):
        return "x".join(map(str, [*shape, dtype]))
    else:
        return f"1x{dtype}"


def stringify_tensor_shape(tensor: Any) -> str:
    if isinstance(tensor, torch.Tensor):
        return stringify_shape(tensor.shape, tensor.dtype)
    if isinstance(tensor, float):
        return "1xf32"
    if isinstance(tensor, int):
        return "1xi32"
    if isinstance(tensor, bool):
        return "1xbool"
