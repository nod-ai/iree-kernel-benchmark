import os
import logging
from pathlib import Path
import argparse
from kernel_bench.kernels.gemm.gemm_utils import GemmConfig

from kernel_bench.core.runner import BenchmarkRunner
from kernel_bench.utils.bench_utils import BenchmarkResult
from kernel_bench.core.base import BENCHMARKS
from kernel_bench.utils.print_utils import get_logger

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json


def create_gemm_plot(
    results: list[BenchmarkResult],
    save_path: str = None,
    backend_names: list[str] = None,
    problems: list[tuple] = None,
):
    if len(results) == 0:
        return

    results = [result for result in results if result.ok]

    def hash_result(result: BenchmarkResult) -> str:
        shape = result.shape
        return f'{shape["M"]}x{shape["N"]}x{shape["K"]}'

    # Get all backends and shapes from results
    result_backends = list(set([result.backend for result in results]))
    backend_shapes = {backend: set() for backend in result_backends}

    for result in results:
        backend_shapes[result.backend].add(hash_result(result))

    # Find common shapes across all backends
    common_shapes = backend_shapes[result_backends[0]]
    for shape_set in backend_shapes.values():
        common_shapes = common_shapes.intersection(shape_set)

    # Build data dictionary
    data = {shape: {} for shape in common_shapes}
    for result in results:
        shape = hash_result(result)
        if shape in common_shapes:
            data[shape][result.backend] = result.tflops

    # Use provided ordering if available, otherwise fall back to discovered order
    if backend_names:
        # Filter to only backends that have results and maintain order
        ordered_backends = [
            b
            for b in backend_names
            if b in result_backends and any(b in data[shape] for shape in common_shapes)
        ]
    else:
        ordered_backends = result_backends

    if problems:
        # Convert problems to hash format and filter to only those with results
        problem_hashes = [f"{p[0]}x{p[1]}x{p[2]}" for p in problems]
        ordered_problems = [h for h in problem_hashes if h in common_shapes]
    else:
        ordered_problems = list(common_shapes)

    sns.set_palette("colorblind")

    # Extract data from the data dictionary
    num_backends = len(ordered_backends)

    # Set up the plot
    x = np.arange(len(ordered_problems))
    width = 0.8 / num_backends  # Adjust width based on number of backends
    fig, ax = plt.subplots(figsize=(max(12, len(ordered_problems) * 0.8), 8))
    plt.rcParams["font.size"] = 14

    # Create bars for each backend
    bars_dict = {}
    colors = plt.cm.Set1(np.linspace(0, 1, num_backends))

    for i, backend in enumerate(ordered_backends):
        # Extract TFLOPs values for this backend across all shapes
        backend_tflops = []
        for shape in ordered_problems:
            if backend in data[shape]:
                backend_tflops.append(data[shape][backend])
            else:
                backend_tflops.append(0)  # Handle missing data

        # Calculate bar position offset
        offset = (i - (num_backends - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            backend_tflops,
            width,
            label=backend.capitalize(),
            color=colors[i],
        )
        bars_dict[backend] = bars

    # Customize the plot
    ax.set_xlabel(f"Problem Configuration (MxNxK)", fontsize=14)
    ax.set_ylabel("TFLOPs", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_problems, rotation=45, ha="right")
    ax.legend()

    # Add value labels on top of bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label bars with data
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # Add labels for all backends
    for bars in bars_dict.values():
        autolabel(bars)

    # Add grid for better readability
    ax.grid(axis="y", alpha=0.3)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        get_logger().info(f"Comparison plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    os.environ["WAVE_CACHE_ON"] = "0"

    parser = argparse.ArgumentParser(description="Benchmarking GEMMs.")
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
        required=True,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--dtype",
        choices=["f16", "bf16", "f8"],
        default="f16",
        help=("Input datatype for kernels (eg: f16, bf16, f8, etc.)\n",),
    )
    parser.add_argument(
        "--variant",
        choices=["NT", "TN", "NN"],
        default="NT",
        help=("Transpose variant for GEMM\n",),
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help=(
            "Backend to run kernels (eg: wave, iree, torch, all, etc.)\n",
            "Multiple backends can be specified as a comma-separated list",
        ),
    )
    parser.add_argument(
        "--problems",
        type=str,
        required=True,
        help=(
            "Comma-separated list of GEMM problems (512x512x512,1024x1024x1024,...)"
            "or path to JSON file containing configurations"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate kernels for numerical accuracy before benchmarking.",
    )

    args = parser.parse_args()
    logger = get_logger()

    dtype = str(args.dtype)
    device = str(args.device)
    variant = str(args.variant)

    try:
        problems = [problem.strip() for problem in str(args.problems).split(",")]
        problem_shapes = [tuple(map(int, problem.split("x"))) for problem in problems]
        configs = [
            (
                "gemm",
                GemmConfig(
                    problem[0],
                    problem[1],
                    problem[2],
                    variant[0],
                    variant[1],
                    dtype,
                ),
            )
            for problem in problem_shapes
        ]
    except:
        logger.error("Failed to parse problems")
        exit(1)

    backend_names = [backend.strip() for backend in str(args.backend).split(",")]
    if "all" in backend_names:
        backend_names = set()
        for kernel_benches in BENCHMARKS.values():
            backend_names.update(kernel_benches.keys())
        backend_names = list(set(backend_names))

    kernel_dir = Path("results/kernels")
    kernel_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for backend_name in backend_names:
        if backend_name not in BENCHMARKS["gemm"]:
            logger.error(
                f"Backend {backend_name} is currently unsupported for GEMMs. Exiting..."
            )
            continue

        logger.info(f"Running GEMM benchmarks for backend {backend_name}")

        bench = BenchmarkRunner(
            backend=backend_name,
            kernel_type="gemm",
            device=device,
            machine=args.machine,
            configs=configs,
            kernel_dir=kernel_dir,
            dump_dir=None,
            debug=True,
            num_iterations=50,
            title=None,
        )
        logger.info(
            f"Generated {len(bench.configs)} GEMM configs for backend {backend_name}."
        )

        results.extend(bench.benchmark_kernels(validate_numerics=args.validate))

    create_gemm_plot(results, args.plot, backend_names, problem_shapes)
