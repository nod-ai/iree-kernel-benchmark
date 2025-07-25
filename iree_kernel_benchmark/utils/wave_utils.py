import torch
import wave_lang.kernel.lang as tkl
from wave_lang.kernel._support.indexing import index_symbol


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


def dtype_to_torch(dtype: str):
    dtype = dtype.lower().strip().replace("_", "")
    return DTYPE_TO_TORCH[dtype]


class TuningSpec:
    def hyperparams(self) -> dict[tkl.IndexSymbol, int]:
        return {index_symbol(attr): val for attr, val in self.__dict__.items()}
