from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import time

import numpy as np
from tqdm import tqdm
from kernel_bench.core.base import create_benchmark
from kernel_bench.core.template import WaveKernelBenchmark, batch_compile_iree_benches
from kernel_bench.kernels.attention import get_default_attention_configs
from kernel_bench.kernels.attention.attention_config import AttentionConfigBMNK
from kernel_bench.kernels.attention.backends.wave_attention import (
    WaveAttentionMHABenchmark,
)
from kernel_bench.tuning.hyperparam.paradigm.tree import TreeParameter
from kernel_bench.tuning.hyperparam.parameters import CategoricalBounds

shape = AttentionConfigBMNK(B=8, M=8192, N=128, K1=128, K2=8192, dtype="f16")

base_bench = WaveAttentionMHABenchmark(
    tag="test",
    backend="wave",
    kernel_type="attention",
    machine="mi300x",
    config=shape,
    kernel_dir=Path("results/kernels"),
)

start_time = time.time()
base_bench.run_bench("hip")
end_time = time.time()
base_runtime = end_time - start_time

print(f"Base runtime: {base_runtime:.4f} seconds")

base_bench.tuning_spec.set_parameter("BLOCK_B", 5)
base_bench.tuning_spec.set_parameter("BLOCK_M", 32)
base_bench.tuning_spec.set_parameter("BLOCK_N", 128)
base_bench.tuning_spec.set_parameter("BLOCK_K2", 32)

base_bench.run_bench("hip")

# 84480 - 64000
