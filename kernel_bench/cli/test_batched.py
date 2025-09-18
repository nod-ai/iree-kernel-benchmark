from itertools import product
import json
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm
from kernel_bench.core.template import WaveKernelBenchmark, batch_compile_iree_benches
from kernel_bench.kernels.attention import get_default_attention_configs
from kernel_bench.kernels.attention.attention_config import AttentionConfigBMNK
from kernel_bench.kernels.attention.backends.wave_attention import (
    WaveAttentionMHABenchmark,
)
from kernel_bench.tuning.hyperparam.paradigm.tree import TreeParameter
from kernel_bench.tuning.hyperparam.parameters import CategoricalBounds

shapes = get_default_attention_configs("attention", "wave")
random.shuffle(shapes)

benches: list[WaveKernelBenchmark] = []
configs: list[dict[str, int]] = []

for tag, shape in tqdm(shapes, desc="Getting candidates for all shapes & configs"):
    base_bench = WaveAttentionMHABenchmark(
        tag="test",
        backend="wave",
        kernel_type="attention",
        machine="mi300x",
        config=shape,
        kernel_dir=Path("results/kernels"),
    )

    params = [
        TreeParameter(p.name, p.bounds.get_range())
        for p in base_bench.tuning_spec.params()
        if not isinstance(p.bounds, CategoricalBounds)
    ]

    candidates = [param.get_candidates(4) for param in params]
    # print("Candidates:", candidates)
    param_vals = product(*candidates)

    for config in param_vals:
        bench = WaveAttentionMHABenchmark(
            tag="test",
            backend="wave",
            kernel_type="attention",
            machine="mi300x",
            config=shape,
            kernel_dir=Path("results/kernels"),
        )
        for i, pval in enumerate(config):
            param = params[i]
            bench.tuning_spec.set_parameter(param.name, pval)

        sat, violated = bench.validate_constraints()
        if not sat:
            # print(f"Violated constraint: {violated}")
            continue

        config = {params[i].name: config[i] for i in range(len(config))}

        benches.append(bench)
        configs.append(config)

results = batch_compile_iree_benches(benches, verbose=True, unique_id=True)

json_res = []

we_chilling = True

for bench, tuning_config, result in zip(benches, configs, results):
    op_config, vmfb_path, success = result

    obj = {
        "kernel": op_config.to_dict(),
        "spec": tuning_config,
        "success": success,
        "expected_mem": list(bench.validate_constraints()[1].values())[0] + 65536,
        "mlir_path": str(vmfb_path.absolute()).replace("vmfb", "mlir"),
    }

    if not success:
        we_chilling = False

    json_res.append(obj)

with open("test.json", "w") as file:
    json.dump(json_res, file)

print("Success" if we_chilling else "Failure")
