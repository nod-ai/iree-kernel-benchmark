from pathlib import Path
from .attentionbench.problems import get_attention_configs, get_attention_configs_gqa
from .gemmbench.problems import get_gemm_configs
from .convbench.problems import get_tk_conv_configs
from dataclasses import asdict
from uuid import uuid4
import json

all_problems = []

all_problems += [
    {
        "id": str(uuid4()),
        "kernelType": "conv",
        "name": problem.get_name(),
        "tag": tag,
        "dtype": problem.input_dtype,
        "allowedBackends": ["iree", "wave", "torch"],
        "problem": asdict(problem),
    }
    for tag, problem in get_tk_conv_configs()
]

all_problems += [
    {
        "id": str(uuid4()),
        "kernelType": "gemm",
        "name": problem.get_name(),
        "tag": tag,
        "dtype": problem.operand_element_type,
        "allowedBackends": ["iree", "torch"]
        + (["wave"] if problem.tA + problem.tB == "NT" else []),
        "problem": asdict(problem),
    }
    for tag, problem in get_gemm_configs("f16", "wave", False)
]

all_problems += [
    {
        "id": str(uuid4()),
        "kernelType": "attention",
        "name": problem.get_name(),
        "tag": tag,
        "dtype": problem.dtype,
        "allowedBackends": ["iree"]
        + (["torch", "wave"] if problem.dtype == "f16" else []),
        "problem": asdict(problem.to_bmnk1k2()),
    }
    for tag, problem in get_attention_configs(use_fp8=True)
]

all_problems += [
    {
        "id": str(uuid4()),
        "kernelType": "attention",
        "name": problem.get_name(),
        "tag": tag,
        "dtype": problem.dtype,
        "allowedBackends": ["wavegqa"],
        "problem": asdict(problem.to_bshd()),
    }
    for tag, problem in get_attention_configs_gqa()
]

result_path = Path("results/configs.json")
with open(result_path, "w") as file:
    json.dump(all_problems, file, indent=4)
print(f"Saved {len(all_problems)} configs to {result_path}")
