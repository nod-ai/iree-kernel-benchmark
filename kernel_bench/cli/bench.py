import os
import logging
from pathlib import Path
import argparse
from kernel_bench.utils.device_utils import HIP_TARGETS
from wave_lang.kernel.wave.constraints import MMAType

from kernel_bench.core.runner import BenchmarkRunner
from kernel_bench.utils.bench_utils import OpConfig, load_configs
from kernel_bench.core.base import LOAD_PROBLEMS, BENCHMARKS, CONFIG_CLASSES
from kernel_bench.utils.print_utils import get_logger

if __name__ == "__main__":
    os.environ["WAVE_CACHE_ON"] = "0"

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
        choices=list(HIP_TARGETS.keys()),
        type=str,
        required=True,
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
        default=50,
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate kernels for numerical accuracy before benchmarking.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    logger = get_logger()

    kernel_types = str(args.kernel_type).split(",")
    if "all" in kernel_types:
        kernel_types = list(BENCHMARKS.keys())

    backend_names = str(args.backend).split(",")
    if "all" in backend_names:
        backend_names = set()
        for kernel_benches in BENCHMARKS.values():
            backend_names.update(kernel_benches.keys())
        backend_names = list(set(backend_names))

    for kernel_type in kernel_types:
        if kernel_type not in BENCHMARKS:
            logger.error(
                f"Kernel type {kernel_type} is currently unsupported. Skipping..."
            )

        if kernel_type == "extend_attention":
            configs = LOAD_PROBLEMS[kernel_type](kernel_type, "wave")
            configs[0][1].get_inputs()

        for backend_name in backend_names:
            if backend_name not in BENCHMARKS[kernel_type]:
                continue

            logger.info(f"Running {kernel_type} benchmarks for backend {backend_name}")

            configs: list[tuple[str, OpConfig]] = []
            if args.load_problems:
                configs = load_configs(
                    args.load_problems,
                    kernel_type,
                    backend_name,
                    CONFIG_CLASSES[kernel_type],
                )
                if args.tune and len(configs) == 0:
                    exit(0)

            if len(configs) == 0:
                configs = LOAD_PROBLEMS[kernel_type](kernel_type, backend_name)

            kernel_dir = Path("results/kernels")
            kernel_dir.mkdir(parents=True, exist_ok=True)
            dump_dir = Path(args.dump_dir) if args.dump_dir else None
            device = args.device

            if kernel_type not in BENCHMARKS:
                logger.error(f"Kernel Type {kernel_type} is currently unsupported.")
                exit(1)

            if backend_name not in BENCHMARKS[kernel_type]:
                logger.error(
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
                title=args.title,
                max_kernels=args.max_kernels,
            )
            logger.info(
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
                bench.benchmark_kernels(validate_numerics=args.validate)
