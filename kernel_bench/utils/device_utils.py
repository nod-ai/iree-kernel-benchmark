from typing import Any, Iterable, Optional
import torch

from kernel_bench.utils.print_utils import get_logger

DTYPE_TO_TORCH = {
    "bf16": torch.bfloat16,
    "f8e5m2": torch.float8_e5m2,
    "f8e5m2fnuz": torch.float8_e5m2fnuz,
    "f8e4m3fn": torch.float8_e4m3fn,
    "f8e4m3fnuz": torch.float8_e4m3fnuz,
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
    "f8e5m2": 8,
    "f8e5m2fnuz": 8,
    "f8e4m3fn": 8,
    "f8e4m3fnuz": 8,
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
    dtype = dtype.lower().strip().replace("_", "")

    if dtype == "f8":
        if not target:
            target = get_hip_target()
        if target == "gfx950":
            dtype = "f8e4m3fn"
        else:
            dtype = "f8e4m3fnuz"

    return dtype


def dtype_to_torch(dtype: str, target: Optional[str] = None):
    dtype = get_device_specific_dtype(dtype, target)
    return DTYPE_TO_TORCH[dtype]


def dtype_to_bits(dtype: str, target: Optional[str] = None):
    dtype = get_device_specific_dtype(dtype, target)
    return DTYPE_TO_BITS[dtype]


def dtype_to_bytes(dtype: str, target: Optional[str] = None):
    return max(1, dtype_to_bits(dtype, target) // 8)


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
