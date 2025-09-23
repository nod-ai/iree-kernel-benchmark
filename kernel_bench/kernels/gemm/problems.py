# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pprint import pprint
import pandas as pd

from kernel_bench.utils.device_utils import dtype_to_bytes
from .gemm_utils import GemmConfig, kDynamic

import re

from kernel_bench.utils.clustering import KernelConfigurationClustering


def get_default_accumulator_element_type(operand_element_type: str) -> str:
    return {
        "f16": "f32",
        "bf16": "f32",
        "f32": "f32",
        "f8E4M3FNUZ": "f32",
        "f8E5M2FNUZ": "f32",
        "f8E4M3FN": "f32",
        "f8E5M2": "f32",
        "i8": "i32",
        "i32": "i32",
    }[operand_element_type]


def get_default_result_element_type(
    operand_element_type: str, raw_accumulators: bool
) -> str:
    return (
        get_default_accumulator_element_type(operand_element_type)
        if raw_accumulators
        else operand_element_type
    )


def is_compute_bound(
    M: int, N: int, K: int, dtype: str, raw_accumulators: bool
) -> bool:
    """Is this GEMM compute (or memory) bound?"""
    magic_ratio = 64
    flops = 2 * M * N * K
    elem_type_bytes = dtype_to_bytes(dtype)
    result_bytes = dtype_to_bytes(
        get_default_result_element_type(dtype, raw_accumulators)
    )
    bytes = elem_type_bytes * (M * K + K * N) + result_bytes * (M * N)
    return flops > magic_ratio * bytes


LLAMA = [
    (32000, 1, 8192, "70b"),
    (10240, 1, 8192, "70b"),
    (8192, 1, 8192, "70b"),
    (57344, 1, 8192, "70b"),
    (8192, 1, 28672, "70b"),
    (10240, 512, 8192, "70b"),
    (8192, 512, 8192, "70b"),
    (57344, 512, 8192, "70b"),
    (8192, 512, 28672, "70b"),
    (10240, 1024, 8192, "70b"),
    (8192, 1024, 8192, "70b"),
    (57344, 1024, 8192, "70b"),
    (8192, 1024, 28672, "70b"),
    (10240, 2048, 8192, "70b"),
    (8192, 2048, 8192, "70b"),
    (57344, 2048, 8192, "70b"),
    (8192, 2048, 28672, "70b"),
    (10240, 3072, 8192, "70b"),
    (8192, 3072, 8192, "70b"),
    (57344, 3072, 8192, "70b"),
    (8192, 3072, 28672, "70b"),
    (10240, 4096, 8192, "70b"),
    (8192, 4096, 8192, "70b"),
    (57344, 4096, 8192, "70b"),
    (8192, 4096, 28672, "70b"),
    (10240, 5120, 8192, "70b"),
    (8192, 5120, 8192, "70b"),
    (57344, 5120, 8192, "70b"),
    (8192, 5120, 28672, "70b"),
    (10240, 6144, 8192, "70b"),
    (8192, 6144, 8192, "70b"),
    (57344, 6144, 8192, "70b"),
    (8192, 6144, 28672, "70b"),
    (10240, 7168, 8192, "70b"),
    (8192, 7168, 8192, "70b"),
    (57344, 7168, 8192, "70b"),
    (8192, 7168, 28672, "70b"),
    (10240, 8192, 8192, "70b"),
    (8192, 8192, 8192, "70b"),
    (57344, 8192, 8192, "70b"),
    (8192, 8192, 28672, "70b"),
    (10240, 9216, 8192, "70b"),
    (8192, 9216, 8192, "70b"),
    (57344, 9216, 8192, "70b"),
    (8192, 9216, 28672, "70b"),
    (10240, 10240, 8192, "70b"),
    (8192, 10240, 8192, "70b"),
    (57344, 10240, 8192, "70b"),
    (8192, 10240, 28672, "70b"),
    (10240, 11264, 8192, "70b"),
    (8192, 11264, 8192, "70b"),
    (57344, 11264, 8192, "70b"),
    (8192, 11264, 28672, "70b"),
    (10240, 12288, 8192, "70b"),
    (8192, 12288, 8192, "70b"),
    (57344, 12288, 8192, "70b"),
    (8192, 12288, 28672, "70b"),
    (10240, 13312, 8192, "70b"),
    (8192, 13312, 8192, "70b"),
    (57344, 13312, 8192, "70b"),
    (8192, 13312, 28672, "70b"),
    (10240, 14336, 8192, "70b"),
    (8192, 14336, 8192, "70b"),
    (57344, 14336, 8192, "70b"),
    (8192, 14336, 28672, "70b"),
    (10240, 15360, 8192, "70b"),
    (8192, 15360, 8192, "70b"),
    (57344, 15360, 8192, "70b"),
    (8192, 15360, 28672, "70b"),
    (10240, 16384, 8192, "70b"),
    (8192, 16384, 8192, "70b"),
    (57344, 16384, 8192, "70b"),
    (8192, 16384, 28672, "70b"),
    (16000, 1, 8192, "70b"),
    (5120, 1, 8192, "70b"),
    (8192, 1, 4096, "70b"),
    (28672, 1, 8192, "70b"),
    (8192, 1, 14336, "70b"),
    (5120, 512, 8192, "70b"),
    (8192, 512, 4096, "70b"),
    (28672, 512, 8192, "70b"),
    (8192, 512, 14336, "70b"),
    (5120, 1024, 8192, "70b"),
    (8192, 1024, 4096, "70b"),
    (28672, 1024, 8192, "70b"),
    (8192, 1024, 14336, "70b"),
    (5120, 2048, 8192, "70b"),
    (8192, 2048, 4096, "70b"),
    (28672, 2048, 8192, "70b"),
    (8192, 2048, 14336, "70b"),
    (5120, 3072, 8192, "70b"),
    (8192, 3072, 4096, "70b"),
    (28672, 3072, 8192, "70b"),
    (8192, 3072, 14336, "70b"),
    (5120, 4096, 8192, "70b"),
    (8192, 4096, 4096, "70b"),
    (28672, 4096, 8192, "70b"),
    (8192, 4096, 14336, "70b"),
    (5120, 5120, 8192, "70b"),
    (8192, 5120, 4096, "70b"),
    (28672, 5120, 8192, "70b"),
    (8192, 5120, 14336, "70b"),
    (5120, 6144, 8192, "70b"),
    (8192, 6144, 4096, "70b"),
    (28672, 6144, 8192, "70b"),
    (8192, 6144, 14336, "70b"),
    (5120, 7168, 8192, "70b"),
    (8192, 7168, 4096, "70b"),
    (28672, 7168, 8192, "70b"),
    (8192, 7168, 14336, "70b"),
    (5120, 8192, 8192, "70b"),
    (8192, 8192, 4096, "70b"),
    (28672, 8192, 8192, "70b"),
    (8192, 8192, 14336, "70b"),
    (5120, 9216, 8192, "70b"),
    (8192, 9216, 4096, "70b"),
    (28672, 9216, 8192, "70b"),
    (8192, 9216, 14336, "70b"),
    (5120, 10240, 8192, "70b"),
    (8192, 10240, 4096, "70b"),
    (28672, 10240, 8192, "70b"),
    (8192, 10240, 14336, "70b"),
    (5120, 11264, 8192, "70b"),
    (8192, 11264, 4096, "70b"),
    (28672, 11264, 8192, "70b"),
    (8192, 11264, 14336, "70b"),
    (5120, 12288, 8192, "70b"),
    (8192, 12288, 4096, "70b"),
    (28672, 12288, 8192, "70b"),
    (8192, 12288, 14336, "70b"),
    (5120, 13312, 8192, "70b"),
    (8192, 13312, 4096, "70b"),
    (28672, 13312, 8192, "70b"),
    (8192, 13312, 14336, "70b"),
    (5120, 14336, 8192, "70b"),
    (8192, 14336, 4096, "70b"),
    (28672, 14336, 8192, "70b"),
    (8192, 14336, 14336, "70b"),
    (5120, 15360, 8192, "70b"),
    (8192, 15360, 4096, "70b"),
    (28672, 15360, 8192, "70b"),
    (8192, 15360, 14336, "70b"),
    (5120, 16384, 8192, "70b"),
    (8192, 16384, 4096, "70b"),
    (28672, 16384, 8192, "70b"),
    (8192, 16384, 14336, "70b"),
    (8000, 1, 8192, "70b"),
    (2560, 1, 8192, "70b"),
    (8192, 1, 2048, "70b"),
    (14336, 1, 8192, "70b"),
    (8192, 1, 7168, "70b"),
    (2560, 512, 8192, "70b"),
    (8192, 512, 2048, "70b"),
    (14336, 512, 8192, "70b"),
    (8192, 512, 7168, "70b"),
    (2560, 1024, 8192, "70b"),
    (8192, 1024, 2048, "70b"),
    (14336, 1024, 8192, "70b"),
    (8192, 1024, 7168, "70b"),
    (2560, 2048, 8192, "70b"),
    (8192, 2048, 2048, "70b"),
    (14336, 2048, 8192, "70b"),
    (8192, 2048, 7168, "70b"),
    (2560, 3072, 8192, "70b"),
    (8192, 3072, 2048, "70b"),
    (14336, 3072, 8192, "70b"),
    (8192, 3072, 7168, "70b"),
    (2560, 4096, 8192, "70b"),
    (8192, 4096, 2048, "70b"),
    (14336, 4096, 8192, "70b"),
    (8192, 4096, 7168, "70b"),
    (2560, 5120, 8192, "70b"),
    (8192, 5120, 2048, "70b"),
    (14336, 5120, 8192, "70b"),
    (8192, 5120, 7168, "70b"),
    (2560, 6144, 8192, "70b"),
    (8192, 6144, 2048, "70b"),
    (14336, 6144, 8192, "70b"),
    (8192, 6144, 7168, "70b"),
    (2560, 7168, 8192, "70b"),
    (8192, 7168, 2048, "70b"),
    (14336, 7168, 8192, "70b"),
    (8192, 7168, 7168, "70b"),
    (2560, 8192, 8192, "70b"),
    (8192, 8192, 2048, "70b"),
    (14336, 8192, 8192, "70b"),
    (8192, 8192, 7168, "70b"),
    (2560, 9216, 8192, "70b"),
    (8192, 9216, 2048, "70b"),
    (14336, 9216, 8192, "70b"),
    (8192, 9216, 7168, "70b"),
    (2560, 10240, 8192, "70b"),
    (8192, 10240, 2048, "70b"),
    (14336, 10240, 8192, "70b"),
    (8192, 10240, 7168, "70b"),
    (2560, 11264, 8192, "70b"),
    (8192, 11264, 2048, "70b"),
    (14336, 11264, 8192, "70b"),
    (8192, 11264, 7168, "70b"),
    (2560, 12288, 8192, "70b"),
    (8192, 12288, 2048, "70b"),
    (14336, 12288, 8192, "70b"),
    (8192, 12288, 7168, "70b"),
    (2560, 13312, 8192, "70b"),
    (8192, 13312, 2048, "70b"),
    (14336, 13312, 8192, "70b"),
    (8192, 13312, 7168, "70b"),
    (2560, 14336, 8192, "70b"),
    (8192, 14336, 2048, "70b"),
    (14336, 14336, 8192, "70b"),
    (8192, 14336, 7168, "70b"),
    (2560, 15360, 8192, "70b"),
    (8192, 15360, 2048, "70b"),
    (14336, 15360, 8192, "70b"),
    (8192, 15360, 7168, "70b"),
    (2560, 16384, 8192, "70b"),
    (8192, 16384, 2048, "70b"),
    (14336, 16384, 8192, "70b"),
    (8192, 16384, 7168, "70b"),
    (4000, 1, 8192, "70b"),
    (1280, 1, 8192, "70b"),
    (8192, 1, 1024, "70b"),
    (7168, 1, 8192, "70b"),
    (8192, 1, 3584, "70b"),
    (1280, 512, 8192, "70b"),
    (8192, 512, 1024, "70b"),
    (7168, 512, 8192, "70b"),
    (8192, 512, 3584, "70b"),
    (1280, 1024, 8192, "70b"),
    (8192, 1024, 1024, "70b"),
    (7168, 1024, 8192, "70b"),
    (8192, 1024, 3584, "70b"),
    (1280, 2048, 8192, "70b"),
    (8192, 2048, 1024, "70b"),
    (7168, 2048, 8192, "70b"),
    (8192, 2048, 3584, "70b"),
    (1280, 3072, 8192, "70b"),
    (8192, 3072, 1024, "70b"),
    (7168, 3072, 8192, "70b"),
    (8192, 3072, 3584, "70b"),
    (1280, 4096, 8192, "70b"),
    (8192, 4096, 1024, "70b"),
    (7168, 4096, 8192, "70b"),
    (8192, 4096, 3584, "70b"),
    (1280, 5120, 8192, "70b"),
    (8192, 5120, 1024, "70b"),
    (7168, 5120, 8192, "70b"),
    (8192, 5120, 3584, "70b"),
    (1280, 6144, 8192, "70b"),
    (8192, 6144, 1024, "70b"),
    (7168, 6144, 8192, "70b"),
    (8192, 6144, 3584, "70b"),
    (1280, 7168, 8192, "70b"),
    (8192, 7168, 1024, "70b"),
    (7168, 7168, 8192, "70b"),
    (8192, 7168, 3584, "70b"),
    (1280, 8192, 8192, "70b"),
    (8192, 8192, 1024, "70b"),
    (7168, 8192, 8192, "70b"),
    (8192, 8192, 3584, "70b"),
    (1280, 9216, 8192, "70b"),
    (8192, 9216, 1024, "70b"),
    (7168, 9216, 8192, "70b"),
    (8192, 9216, 3584, "70b"),
    (1280, 10240, 8192, "70b"),
    (8192, 10240, 1024, "70b"),
    (7168, 10240, 8192, "70b"),
    (8192, 10240, 3584, "70b"),
    (1280, 11264, 8192, "70b"),
    (8192, 11264, 1024, "70b"),
    (7168, 11264, 8192, "70b"),
    (8192, 11264, 3584, "70b"),
    (1280, 12288, 8192, "70b"),
    (8192, 12288, 1024, "70b"),
    (7168, 12288, 8192, "70b"),
    (8192, 12288, 3584, "70b"),
    (1280, 13312, 8192, "70b"),
    (8192, 13312, 1024, "70b"),
    (7168, 13312, 8192, "70b"),
    (8192, 13312, 3584, "70b"),
    (1280, 14336, 8192, "70b"),
    (8192, 14336, 1024, "70b"),
    (7168, 14336, 8192, "70b"),
    (8192, 14336, 3584, "70b"),
    (1280, 15360, 8192, "70b"),
    (8192, 15360, 1024, "70b"),
    (7168, 15360, 8192, "70b"),
    (8192, 15360, 3584, "70b"),
    (1280, 16384, 8192, "70b"),
    (8192, 16384, 1024, "70b"),
    (7168, 16384, 8192, "70b"),
    (8192, 16384, 3584, "70b"),
    (32000, 1, 5120, "13b"),
    (15360, 1, 5120, "13b"),
    (5120, 1, 5120, "13b"),
    (27648, 1, 5120, "13b"),
    (5120, 1, 13824, "13b"),
    (15360, 512, 5120, "13b"),
    (5120, 512, 5120, "13b"),
    (27648, 512, 5120, "13b"),
    (5120, 512, 13824, "13b"),
    (15360, 1024, 5120, "13b"),
    (5120, 1024, 5120, "13b"),
    (27648, 1024, 5120, "13b"),
    (5120, 1024, 13824, "13b"),
    (15360, 2048, 5120, "13b"),
    (5120, 2048, 5120, "13b"),
    (27648, 2048, 5120, "13b"),
    (5120, 2048, 13824, "13b"),
    (15360, 3072, 5120, "13b"),
    (5120, 3072, 5120, "13b"),
    (27648, 3072, 5120, "13b"),
    (5120, 3072, 13824, "13b"),
    (15360, 4096, 5120, "13b"),
    (5120, 4096, 5120, "13b"),
    (27648, 4096, 5120, "13b"),
    (5120, 4096, 13824, "13b"),
    (15360, 5120, 5120, "13b"),
    (5120, 5120, 5120, "13b"),
    (27648, 5120, 5120, "13b"),
    (5120, 5120, 13824, "13b"),
    (15360, 6144, 5120, "13b"),
    (5120, 6144, 5120, "13b"),
    (27648, 6144, 5120, "13b"),
    (5120, 6144, 13824, "13b"),
    (15360, 7168, 5120, "13b"),
    (5120, 7168, 5120, "13b"),
    (27648, 7168, 5120, "13b"),
    (5120, 7168, 13824, "13b"),
    (15360, 8192, 5120, "13b"),
    (5120, 8192, 5120, "13b"),
    (27648, 8192, 5120, "13b"),
    (5120, 8192, 13824, "13b"),
    (15360, 9216, 5120, "13b"),
    (5120, 9216, 5120, "13b"),
    (27648, 9216, 5120, "13b"),
    (5120, 9216, 13824, "13b"),
    (15360, 10240, 5120, "13b"),
    (5120, 10240, 5120, "13b"),
    (27648, 10240, 5120, "13b"),
    (5120, 10240, 13824, "13b"),
    (15360, 11264, 5120, "13b"),
    (5120, 11264, 5120, "13b"),
    (27648, 11264, 5120, "13b"),
    (5120, 11264, 13824, "13b"),
    (15360, 12288, 5120, "13b"),
    (5120, 12288, 5120, "13b"),
    (27648, 12288, 5120, "13b"),
    (5120, 12288, 13824, "13b"),
    (15360, 13312, 5120, "13b"),
    (5120, 13312, 5120, "13b"),
    (27648, 13312, 5120, "13b"),
    (5120, 13312, 13824, "13b"),
    (15360, 14336, 5120, "13b"),
    (5120, 14336, 5120, "13b"),
    (27648, 14336, 5120, "13b"),
    (5120, 14336, 13824, "13b"),
    (15360, 15360, 5120, "13b"),
    (5120, 15360, 5120, "13b"),
    (27648, 15360, 5120, "13b"),
    (5120, 15360, 13824, "13b"),
    (15360, 16384, 5120, "13b"),
    (5120, 16384, 5120, "13b"),
    (27648, 16384, 5120, "13b"),
    (5120, 16384, 13824, "13b"),
    (16000, 1, 5120, "13b"),
    (7680, 1, 5120, "13b"),
    (5120, 1, 2560, "13b"),
    (13824, 1, 5120, "13b"),
    (5120, 1, 6912, "13b"),
    (7680, 512, 5120, "13b"),
    (5120, 512, 2560, "13b"),
    (13824, 512, 5120, "13b"),
    (5120, 512, 6912, "13b"),
    (7680, 1024, 5120, "13b"),
    (5120, 1024, 2560, "13b"),
    (13824, 1024, 5120, "13b"),
    (5120, 1024, 6912, "13b"),
    (7680, 2048, 5120, "13b"),
    (5120, 2048, 2560, "13b"),
    (13824, 2048, 5120, "13b"),
    (5120, 2048, 6912, "13b"),
    (7680, 3072, 5120, "13b"),
    (5120, 3072, 2560, "13b"),
    (13824, 3072, 5120, "13b"),
    (5120, 3072, 6912, "13b"),
    (7680, 4096, 5120, "13b"),
    (5120, 4096, 2560, "13b"),
    (13824, 4096, 5120, "13b"),
    (5120, 4096, 6912, "13b"),
    (7680, 5120, 5120, "13b"),
    (5120, 5120, 2560, "13b"),
    (13824, 5120, 5120, "13b"),
    (5120, 5120, 6912, "13b"),
    (7680, 6144, 5120, "13b"),
    (5120, 6144, 2560, "13b"),
    (13824, 6144, 5120, "13b"),
    (5120, 6144, 6912, "13b"),
    (7680, 7168, 5120, "13b"),
    (5120, 7168, 2560, "13b"),
    (13824, 7168, 5120, "13b"),
    (5120, 7168, 6912, "13b"),
    (7680, 8192, 5120, "13b"),
    (5120, 8192, 2560, "13b"),
    (13824, 8192, 5120, "13b"),
    (5120, 8192, 6912, "13b"),
    (7680, 9216, 5120, "13b"),
    (5120, 9216, 2560, "13b"),
    (13824, 9216, 5120, "13b"),
    (5120, 9216, 6912, "13b"),
    (7680, 10240, 5120, "13b"),
    (5120, 10240, 2560, "13b"),
    (13824, 10240, 5120, "13b"),
    (5120, 10240, 6912, "13b"),
    (7680, 11264, 5120, "13b"),
    (5120, 11264, 2560, "13b"),
    (13824, 11264, 5120, "13b"),
    (5120, 11264, 6912, "13b"),
    (7680, 12288, 5120, "13b"),
    (5120, 12288, 2560, "13b"),
    (13824, 12288, 5120, "13b"),
    (5120, 12288, 6912, "13b"),
    (7680, 13312, 5120, "13b"),
    (5120, 13312, 2560, "13b"),
    (13824, 13312, 5120, "13b"),
    (5120, 13312, 6912, "13b"),
    (7680, 14336, 5120, "13b"),
    (5120, 14336, 2560, "13b"),
    (13824, 14336, 5120, "13b"),
    (5120, 14336, 6912, "13b"),
    (7680, 15360, 5120, "13b"),
    (5120, 15360, 2560, "13b"),
    (13824, 15360, 5120, "13b"),
    (5120, 15360, 6912, "13b"),
    (7680, 16384, 5120, "13b"),
    (5120, 16384, 2560, "13b"),
    (13824, 16384, 5120, "13b"),
    (5120, 16384, 6912, "13b"),
    (8000, 1, 5120, "13b"),
    (3840, 1, 5120, "13b"),
    (5120, 1, 1280, "13b"),
    (6912, 1, 5120, "13b"),
    (5120, 1, 3456, "13b"),
    (3840, 512, 5120, "13b"),
    (5120, 512, 1280, "13b"),
    (6912, 512, 5120, "13b"),
    (5120, 512, 3456, "13b"),
    (3840, 1024, 5120, "13b"),
    (5120, 1024, 1280, "13b"),
    (6912, 1024, 5120, "13b"),
    (5120, 1024, 3456, "13b"),
    (3840, 2048, 5120, "13b"),
    (5120, 2048, 1280, "13b"),
    (6912, 2048, 5120, "13b"),
    (5120, 2048, 3456, "13b"),
    (3840, 3072, 5120, "13b"),
    (5120, 3072, 1280, "13b"),
    (6912, 3072, 5120, "13b"),
    (5120, 3072, 3456, "13b"),
    (3840, 4096, 5120, "13b"),
    (5120, 4096, 1280, "13b"),
    (6912, 4096, 5120, "13b"),
    (5120, 4096, 3456, "13b"),
    (3840, 5120, 5120, "13b"),
    (5120, 5120, 1280, "13b"),
    (6912, 5120, 5120, "13b"),
    (5120, 5120, 3456, "13b"),
    (3840, 6144, 5120, "13b"),
    (5120, 6144, 1280, "13b"),
    (6912, 6144, 5120, "13b"),
    (5120, 6144, 3456, "13b"),
    (3840, 7168, 5120, "13b"),
    (5120, 7168, 1280, "13b"),
    (6912, 7168, 5120, "13b"),
    (5120, 7168, 3456, "13b"),
    (3840, 8192, 5120, "13b"),
    (5120, 8192, 1280, "13b"),
    (6912, 8192, 5120, "13b"),
    (5120, 8192, 3456, "13b"),
    (3840, 9216, 5120, "13b"),
    (5120, 9216, 1280, "13b"),
    (6912, 9216, 5120, "13b"),
    (5120, 9216, 3456, "13b"),
    (3840, 10240, 5120, "13b"),
    (5120, 10240, 1280, "13b"),
    (6912, 10240, 5120, "13b"),
    (5120, 10240, 3456, "13b"),
    (3840, 11264, 5120, "13b"),
    (5120, 11264, 1280, "13b"),
    (6912, 11264, 5120, "13b"),
    (5120, 11264, 3456, "13b"),
    (3840, 12288, 5120, "13b"),
    (5120, 12288, 1280, "13b"),
    (6912, 12288, 5120, "13b"),
    (5120, 12288, 3456, "13b"),
    (3840, 13312, 5120, "13b"),
    (5120, 13312, 1280, "13b"),
    (6912, 13312, 5120, "13b"),
    (5120, 13312, 3456, "13b"),
    (3840, 14336, 5120, "13b"),
    (5120, 14336, 1280, "13b"),
    (6912, 14336, 5120, "13b"),
    (5120, 14336, 3456, "13b"),
    (3840, 15360, 5120, "13b"),
    (5120, 15360, 1280, "13b"),
    (6912, 15360, 5120, "13b"),
    (5120, 15360, 3456, "13b"),
    (3840, 16384, 5120, "13b"),
    (5120, 16384, 1280, "13b"),
    (6912, 16384, 5120, "13b"),
    (5120, 16384, 3456, "13b"),
    (4000, 1, 5120, "13b"),
    (1920, 1, 5120, "13b"),
    (5120, 1, 640, "13b"),
    (3456, 1, 5120, "13b"),
    (5120, 1, 1728, "13b"),
    (1920, 512, 5120, "13b"),
    (5120, 512, 640, "13b"),
    (3456, 512, 5120, "13b"),
    (5120, 512, 1728, "13b"),
    (1920, 1024, 5120, "13b"),
    (5120, 1024, 640, "13b"),
    (3456, 1024, 5120, "13b"),
    (5120, 1024, 1728, "13b"),
    (1920, 2048, 5120, "13b"),
    (5120, 2048, 640, "13b"),
    (3456, 2048, 5120, "13b"),
    (5120, 2048, 1728, "13b"),
    (1920, 3072, 5120, "13b"),
    (5120, 3072, 640, "13b"),
    (3456, 3072, 5120, "13b"),
    (5120, 3072, 1728, "13b"),
    (1920, 4096, 5120, "13b"),
    (5120, 4096, 640, "13b"),
    (3456, 4096, 5120, "13b"),
    (5120, 4096, 1728, "13b"),
    (1920, 5120, 5120, "13b"),
    (5120, 5120, 640, "13b"),
    (3456, 5120, 5120, "13b"),
    (5120, 5120, 1728, "13b"),
    (1920, 6144, 5120, "13b"),
    (5120, 6144, 640, "13b"),
    (3456, 6144, 5120, "13b"),
    (5120, 6144, 1728, "13b"),
    (1920, 7168, 5120, "13b"),
    (5120, 7168, 640, "13b"),
    (3456, 7168, 5120, "13b"),
    (5120, 7168, 1728, "13b"),
    (1920, 8192, 5120, "13b"),
    (5120, 8192, 640, "13b"),
    (3456, 8192, 5120, "13b"),
    (5120, 8192, 1728, "13b"),
    (1920, 9216, 5120, "13b"),
    (5120, 9216, 640, "13b"),
    (3456, 9216, 5120, "13b"),
    (5120, 9216, 1728, "13b"),
    (1920, 10240, 5120, "13b"),
    (5120, 10240, 640, "13b"),
    (3456, 10240, 5120, "13b"),
    (5120, 10240, 1728, "13b"),
    (1920, 11264, 5120, "13b"),
    (5120, 11264, 640, "13b"),
    (3456, 11264, 5120, "13b"),
    (5120, 11264, 1728, "13b"),
    (1920, 12288, 5120, "13b"),
    (5120, 12288, 640, "13b"),
    (3456, 12288, 5120, "13b"),
    (5120, 12288, 1728, "13b"),
    (1920, 13312, 5120, "13b"),
    (5120, 13312, 640, "13b"),
    (3456, 13312, 5120, "13b"),
    (5120, 13312, 1728, "13b"),
    (1920, 14336, 5120, "13b"),
    (5120, 14336, 640, "13b"),
    (3456, 14336, 5120, "13b"),
    (5120, 14336, 1728, "13b"),
    (1920, 15360, 5120, "13b"),
    (5120, 15360, 640, "13b"),
    (3456, 15360, 5120, "13b"),
    (5120, 15360, 1728, "13b"),
    (1920, 16384, 5120, "13b"),
    (5120, 16384, 640, "13b"),
    (3456, 16384, 5120, "13b"),
    (5120, 16384, 1728, "13b"),
    (512, 4096, 14336, "8b_prefill"),
    (512, 14336, 4096, "8b_prefill"),
    (512, 4096, 4096, "8b_prefill"),
    (512, 1024, 4096, "8b_prefill"),
]

GPT4 = [
    (16, 16, 1024),
    (16, 16, 8192),
    (16, 16, 65536),
    (16, 2048, 1024),
    (16, 2048, 8192),
    (16, 2048, 65536),
    (16, 8192, 1024),
    (16, 8192, 8192),
    (16, 8192, 65536),
    (2048, 16, 1024),
    (2048, 16, 8192),
    (2048, 16, 65536),
    (2048, 2048, 1024),
    (2048, 2048, 8192),
    (2048, 2048, 65536),
    (2048, 8192, 1024),
    (2048, 8192, 8192),
    (2048, 8192, 65536),
    (8192, 16, 1024),
    (8192, 16, 8192),
    (8192, 16, 65536),
    (8192, 2048, 1024),
    (8192, 2048, 8192),
    (8192, 2048, 65536),
    (8192, 8192, 1024),
    (8192, 8192, 8192),
    (8192, 8192, 65536),
]

UNET = [
    # UNET
    (2048, 10240, 1280),
    (2048, 1280, 1280),
    (2048, 1280, 5120),
    (128, 1280, 2048),
    (8192, 640, 640),
    (8192, 640, 2560),
    (8192, 5120, 640),
    # PUNET
    (64, 640, 2048),
    (64, 1280, 2048),
    (1024, 1280, 1280),
    (1024, 1280, 5120),
    (1024, 10240, 1280),
    (4096, 640, 640),
    (4096, 640, 2560),
    (4096, 5120, 640),
]

SQUARE = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]


def llama8b_prefill(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    configs = []
    """LLAMA 8b Prefill."""
    for m, n, k, model in LLAMA:
        if model == "8b_prefill":
            configs.append(
                GemmConfig(
                    kDynamic,
                    n,
                    k,
                    "N",
                    "T",
                    dtype,
                    get_default_accumulator_element_type(dtype),
                    get_default_result_element_type(dtype, raw_accumulators),
                    runtime_dim=m,
                )
            )
    return configs


def llama13bmatvec(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    configs = []
    """LLAMA 13b, single batch."""
    for m, n, k, model in LLAMA:
        if n == 1 and model == "13b":
            configs.append(
                GemmConfig(
                    m,
                    n,
                    k,
                    "N",
                    "T",
                    dtype,
                    get_default_accumulator_element_type(dtype),
                    get_default_result_element_type(dtype, raw_accumulators),
                )
            )
    return configs


def llama70bmatvec(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """LLAMA 70b, single batch."""
    configs = []
    for m, n, k, model in LLAMA:
        if n == 1 and model == "70b":
            configs.append(
                GemmConfig(
                    m,
                    n,
                    k,
                    "N",
                    "T",
                    dtype,
                    get_default_accumulator_element_type(dtype),
                    get_default_result_element_type(dtype, raw_accumulators),
                )
            )
    return configs


def llama13bskinny(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """LLAMA 13b, multiple batches."""
    configs = []
    for m, n, k, model in LLAMA:
        if n == 1 and model == "13b":
            for batch in [2, 4, 8, 16, 32]:
                configs.append(
                    GemmConfig(
                        m,
                        batch,
                        k,
                        "N",
                        "T",
                        dtype,
                        get_default_accumulator_element_type(dtype),
                        get_default_result_element_type(dtype, raw_accumulators),
                    )
                )
    return configs


def llama70bskinny(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """LLAMA 70b, multiple batches."""
    configs = []
    for m, n, k, model in LLAMA:
        if n == 1 and model == "70b":
            for batch in [2, 4, 8, 16, 32]:
                configs.append(
                    GemmConfig(
                        m,
                        batch,
                        k,
                        "N",
                        "T",
                        dtype,
                        get_default_accumulator_element_type(dtype),
                        get_default_result_element_type(dtype, raw_accumulators),
                    )
                )
    return configs


def gpt4memory(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """GPT4 memory bound GEMMs."""
    configs = []
    for m, n, k in GPT4:
        hgemm = GemmConfig(
            m,
            n,
            k,
            "N",
            "N",
            dtype,
            get_default_accumulator_element_type(dtype),
            get_default_result_element_type(dtype, raw_accumulators),
        )
        if not is_compute_bound(m, n, k, dtype, raw_accumulators):
            configs.append(hgemm)
    return configs


def gpt4compute(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """GPT4 compute bound GEMMs."""
    configs = []
    for m, n, k in GPT4:
        hgemm = GemmConfig(
            m,
            n,
            k,
            "N",
            "N",
            dtype,
            get_default_accumulator_element_type(dtype),
            get_default_result_element_type(dtype, raw_accumulators),
        )
        if is_compute_bound(m, n, k, dtype, raw_accumulators):
            configs.append(hgemm)
    return configs


def tk_default(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """TK Shapes."""
    acc_type = get_default_accumulator_element_type(dtype)
    res_type = get_default_result_element_type(dtype, raw_accumulators)
    configs = []
    M, N, K = 2048, 10240, 1280
    configs.append(GemmConfig(M, N, K, "N", "T", dtype, acc_type, res_type))
    M, N, K = 2048, 1280, 1280
    configs.append(GemmConfig(M, N, K, "N", "T", dtype, acc_type, res_type))
    M, N, K = 2048, 1280, 5120
    configs.append(GemmConfig(M, N, K, "N", "T", dtype, acc_type, res_type))
    M, N, K = 128, 1280, 2048
    configs.append(GemmConfig(M, N, K, "N", "T", dtype, acc_type, res_type))
    M, N, K = 8192, 5120, 640
    configs.append(GemmConfig(M, N, K, "N", "T", dtype, acc_type, res_type))
    return configs


def tk_unet(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """UNET Shapes for TK."""
    configs = []
    for m, n, k in UNET:
        configs.append(
            GemmConfig(
                m,
                n,
                k,
                "N",
                "T",
                dtype,
                get_default_accumulator_element_type(dtype),
                get_default_result_element_type(dtype, raw_accumulators),
            )
        )
    return configs


def llama70bmemory(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """LLAMA 70b memory bound GEMMs; NT."""
    configs = []
    for n in [1280, 3584, 7168]:
        configs.append(
            GemmConfig(
                2,
                n,
                8192,
                "N",
                "T",
                dtype,
                get_default_accumulator_element_type(dtype),
                get_default_result_element_type(dtype, raw_accumulators),
            )
        )
    return configs


def compute(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    """Compute bound GEMMs."""
    configs = []
    for tA, tB in [("N", "N"), ("N", "T"), ("T", "N")]:
        configs.append(
            GemmConfig(
                4096,
                4096,
                8192,
                tA,
                tB,
                dtype,
                get_default_accumulator_element_type(dtype),
                get_default_result_element_type(dtype, raw_accumulators),
            )
        )
    return configs


def unet(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    configs = []
    for tA, tB in [("N", "N"), ("N", "T")]:
        for m, n, k in UNET:
            configs.append(
                GemmConfig(
                    m,
                    n,
                    k,
                    tA,
                    tB,
                    dtype,
                    get_default_accumulator_element_type(dtype),
                    get_default_result_element_type(dtype, raw_accumulators),
                )
            )
    return configs


def square(dtype: str, raw_accumulators: bool) -> list[GemmConfig]:
    configs = []
    for m, n, k in SQUARE:
        configs.append(
            GemmConfig(
                m,
                n,
                k,
                "N",
                "T",
                dtype,
                get_default_accumulator_element_type(dtype),
                get_default_result_element_type(dtype, raw_accumulators),
            )
        )
    return configs


def cai(dtype: str) -> list[GemmConfig]:
    shapes = [
        (16384, 4096, 32768),
        (16384, 32768, 4096),
        (16384, 4096, 8192),
        (16384, 8704, 4096),
        (16384, 8192, 4096),
    ]

    configs = [
        GemmConfig(
            *shape,
            tA="N",
            tB="T",
            operand_element_type=dtype,
            accumulator_element_type=get_default_accumulator_element_type(dtype),
            result_element_type="f32",
        )
        for shape in shapes
    ]

    return configs


def get_b200_gemm_configs(backend: str) -> list[tuple[str, GemmConfig]]:
    def parse_b200_dtype(dtype: str):
        dtype = dtype.split("_")[0]
        if "f8" in dtype and backend == "wave":
            dtype = "f8"
        return dtype

    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_df = pd.read_csv(os.path.join(current_dir, "b200_gemms.csv"))
    configs: list[tuple[str, GemmConfig]] = []

    for index, row in gemm_df.iterrows():
        if row["Batch Count"] > 1:
            continue

        input_dtype = parse_b200_dtype(row["A Type"])
        accumulator_dtype = parse_b200_dtype(row["C Type"])
        result_dtype = parse_b200_dtype(row["D Type"])

        tA = row["Transpose A"]
        tB = row["Transpose B"]

        configs.append(
            (
                row["Subworkload"],
                GemmConfig(
                    M=row["M"],
                    N=row["N"],
                    K=row["K"],
                    tA=tA,
                    tB=tB,
                    operand_element_type=input_dtype,
                    accumulator_element_type=accumulator_dtype,
                    result_element_type=result_dtype,
                ),
            )
        )

    return configs


def get_gemm_configs(
    dtype: str, backend: str, raw_accumulators: bool
) -> list[tuple[str, GemmConfig]]:
    llama8b_prefill_configs = llama8b_prefill(dtype, raw_accumulators)
    llama13bmatvec_configs = llama13bmatvec(dtype, raw_accumulators)
    llama70bmatvec_configs = llama70bmatvec(dtype, raw_accumulators)
    llama13bskinny_configs = llama13bskinny(dtype, raw_accumulators)
    llama70bskinny_configs = llama70bskinny(dtype, raw_accumulators)
    gpt4compute_configs = gpt4compute(dtype, raw_accumulators)
    llama70bmemory_configs = llama70bmemory(dtype, raw_accumulators)
    tk_default_configs = tk_default(dtype, raw_accumulators)
    compute_configs = compute(dtype, raw_accumulators)
    unet_configs = unet(dtype, raw_accumulators)
    square_configs = square(dtype, raw_accumulators)
    cai_configs = cai(dtype)

    all_configs: list[tuple[str, GemmConfig]] = []
    if backend == "iree":
        all_configs += [("llama8b_prefill", x) for x in llama8b_prefill_configs]
    all_configs += [("llama13bmatvec", x) for x in llama13bmatvec_configs]
    all_configs += [("llama70bmatvec", x) for x in llama70bmatvec_configs]
    all_configs += [("llama13bskinny", x) for x in llama13bskinny_configs]
    all_configs += [("llama70bskinny", x) for x in llama70bskinny_configs]
    all_configs += [("gpt4compute", x) for x in gpt4compute_configs]
    all_configs += [("llama70bmemory", x) for x in llama70bmemory_configs]
    all_configs += [("compute", x) for x in compute_configs]
    all_configs += [("unet", x) for x in unet_configs]
    all_configs += [("square", x) for x in square_configs]
    all_configs += [("tk", x) for x in tk_default_configs]
    all_configs += [("cai", x) for x in cai_configs]

    return all_configs


def get_gemm_comparison() -> list[tuple[str, GemmConfig]]:
    return [
        ("comparison", GemmConfig(1024, 1280, 5120, "N", "T", "f16", "f16", "f32")),
        ("comparison", GemmConfig(1024, 10240, 1280, "N", "T", "f16", "f16", "f32")),
        ("comparison", GemmConfig(4096, 640, 640, "N", "T", "f16", "f16", "f32")),
    ]


def get_paper_gemms() -> list[tuple[str, GemmConfig]]:
    shapes = [
        (2048, 10240, 1280),
        (2048, 1280, 1280),
        (2048, 1280, 5120),
        (128, 1280, 2048),
        (8192, 5120, 640),
    ]

    return [
        ("paper", GemmConfig(M, N, K, "N", "T", "f16", "f16", "f32"))
        for M, N, K in shapes
    ]


def get_tk_gemm_configs(
    dtype: str, raw_accumulators: bool
) -> list[tuple[str, GemmConfig]]:
    configs: list[tuple[str, GemmConfig]] = []
    tk_default_configs = tk_default(dtype, raw_accumulators)
    tk_unet_configs = tk_unet(dtype, raw_accumulators)

    configs += [("tk", x) for x in tk_default_configs]
    configs += [("unet", x) for x in tk_unet_configs]
    return configs


def get_matching_configs(
    tagged_configs: list[tuple[str, GemmConfig]],
    variants: list[str],
    tag_regex: str,
    config_regex: str,
) -> list[tuple[str, GemmConfig]]:
    tag_re = re.compile(tag_regex)
    config_re = re.compile(config_regex)
    matching_configs: list[tuple[str, GemmConfig]] = []
    for tag, config in tagged_configs:
        if f"{config.tA}{config.tB}" not in variants:
            continue
        if not tag_re.match(tag):
            continue
        if not config_re.match(config.get_name()):
            continue
        # TODO(https://github.com/iree-org/iree/issues/20446):
        # Mx1xK transpose-A/-B configurations temporarily skipped because they
        # trigger an IREE/MLIR bug causing a compilation failure.
        if config.N == 1 and (config.tA == "T" or config.tB == "T"):
            continue
        matching_configs.append((tag, config))

    return matching_configs
