from multiprocessing import Pool, cpu_count, Manager
import logging
from typing import Literal
from pathlib import Path
import json
import argparse
import sys
from wave_lang.kernel.wave.constraints import MMAType

from .attentionbench.attention_config import AttentionConfigBMNK
from .convbench.conv_utils import ConvConfig
from .gemmbench.gemm_utils import GemmConfig

from .utils import *
from .utils.runner import BenchmarkRunner

from .gemmbench import (
    GEMM_BENCH,
    get_default_gemm_configs,
)
from .attentionbench import (
    ATTENTION_BENCH,
    get_default_attention_configs,
)
from .convbench import (
    CONV_BENCH,
    get_default_conv_configs,
)

BENCHMARKS: dict[str, dict[str, KernelBenchmark]] = {}
BENCHMARKS.update(GEMM_BENCH)
BENCHMARKS.update(ATTENTION_BENCH)
BENCHMARKS.update(CONV_BENCH)

LOAD_PROBLEMS = {
    "gemm": get_default_gemm_configs,
    "attention": get_default_attention_configs,
    "conv": get_default_conv_configs,
}

CONFIG_CLASSES = {
    "gemm": GemmConfig,
    "attention": AttentionConfigBMNK,
    "conv": ConvConfig,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level",
    )
    parser.add_argument(
        "--device",
        help="The IREE device to execute benchmarks on",
        type=str,
        default="hip",
    )
    parser.add_argument(
        "--machine",
        help="Machine used for benchmarking (ex: mi300x, mi325x, etc.).",
        type=str,
        default="mi325x",
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--kernel_type",
        type=str,
        required=True,
        help="Kernel Type (eg: attention, gemm, conv, etc.)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Backend to run kernels (eg: wave, iree, torch, etc.)",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="Directory to which executable files will be dumped.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Uses heuristic approach to optimize mfma variant, tiling, and waves.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of tuning trials.",
    )
    parser.add_argument(
        "--use_tuned",
        type=str,
        default=None,
        help="Path to json file with tuned results.",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Title of run for save path"
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=None,
        help="Maximum number of kernels to benchmark.",
    )
    parser.add_argument(
        "--load_problems", type=str, default=None, help="Path to custom problem list."
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    kernel_type = str(args.kernel_type)
    backend_name = str(args.backend)

    mfma_config = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x16_K8_F16)

    configs: list[tuple[str, OpConfig]] = []
    if args.load_problems:
        configs = load_configs(
            args.load_problems, kernel_type, backend_name, CONFIG_CLASSES[kernel_type]
        )
        if args.tune and len(configs) == 0:
            exit(0)

    if len(configs) == 0:
        configs = LOAD_PROBLEMS[kernel_type](kernel_type, backend_name)

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "kernels"
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    device = args.device
    kernel_dir.mkdir(parents=True, exist_ok=True)

    if kernel_type not in BENCHMARKS:
        print(f"Kernel Type {kernel_type} is currently unsupported.")
        exit(1)

    if backend_name not in BENCHMARKS[kernel_type]:
        print(
            f"Backend {backend_name} is currently unsupported for {kernel_type} benchmarking."
        )
        exit(1)

    bench = BenchmarkRunner(
        backend=backend_name,
        kernel_type=kernel_type,
        device=device,
        machine=args.machine,
        configs=configs,
        kernel_dir=kernel_dir,
        dump_dir=dump_dir,
        debug=True,
        num_iterations=args.iterations,
    )
    bench.reduce_configs(args.max_kernels)
    print(
        f"Generated {len(bench.configs)} {kernel_type} configs for backend {backend_name}."
    )

    if args.tune:
        bench.tune_kernels(
            num_trials=args.num_trials,
        )
        # bench.tune_scheduling(max_iterations=args.num_trials)

    else:
        if args.use_tuned:
            bench.load_tuned_results(args.use_tuned)
        bench.benchmark_kernels()
