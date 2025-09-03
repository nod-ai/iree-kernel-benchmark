from dataclasses import asdict
import torch
import sympy
from typing import Any, Dict, List, Optional, Tuple, Type, override

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


def dtype_to_torch(dtype: str):
    dtype = dtype.lower().strip().replace("_", "")
    return DTYPE_TO_TORCH[dtype]


def dtype_to_bits(dtype: str):
    dtype = dtype.lower().strip().replace("_", "")
    return DTYPE_TO_BITS[dtype]


def dtype_to_bytes(dtype: str):
    return max(1, dtype_to_bits(dtype) // 8)
