import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import csv
import argparse
import sys
import re
from collections import OrderedDict

from wave_lang.kernel.wave.constraints import MMAType
from ..utils import *
from .conv_utils import *
from .problems import get_conv_configs, get_tk_conv_configs, get_conv_test_configs

from .wave_conv import WaveConvBenchmark
from .iree_conv import IREEConvBenchmark

BACKEND_TO_CONV_BENCH = {
    "wave": WaveConvBenchmark,
    "iree": IREEConvBenchmark,
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
        "--Xiree_compile",
        nargs="+",
        default=[],
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`.",
    )
    parser.add_argument(
        "--filter-config",
        help="Only execute configs matching the provided regex",
        type=str,
    )
    parser.add_argument(
        "--roofline",
        help="Comma seperated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--backend",
        choices=["iree", "wave", "wavegqa", "torch"],
        default="iree",
        help="Backend to run kernels",
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
        default=3,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Uses heuristic approach to optimize mfma variant, tiling, and waves.",
    )
    parser.add_argument(
        "--tuning_config",
        type=str,
        default=None,
        help="Path to tuning configuration file.",
    )
    parser.add_argument(
        "--use_tuned",
        type=str,
        default=None,
        help="Path to json file with tuned results.",
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=None,
        help="Maximum number of kernels to benchmark.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline, args.plot, args.batch, args.dtype, args.model)
        sys.exit()

    configs = get_tk_conv_configs()
    if args.filter_config is not None:
        filter_regex = re.compile(args.filter_config)
        configs = list(
            filter(lambda config: filter_regex.match(config[1].get_name()), configs)
        )

    backend_name = args.backend

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "kernels"
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    kernel_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    bench_params = {
        "backend": backend_name,
        "kernel_type": "conv",
        "device": device,
        "configs": configs,
        "kernel_dir": kernel_dir,
        "dump_dir": dump_dir,
        "debug": True,
        "num_iterations": args.iterations,
    }

    bench: KernelBenchmark = BACKEND_TO_CONV_BENCH[backend_name](**bench_params)
    bench.reduce_configs(args.max_kernels)
    print(f"Generated {len(bench.configs)} gemm configs.")

    if args.tune:
        mfma_configs: List[MMAType] = [
            MMAType.F32_16x16x16_F16,
        ]
        tiling_constraints: List[TuningConstraint] = [
            TuningConstraint(name="BLOCK_M", min=16, max=256, step=4),
            TuningConstraint(name="BLOCK_N", min=16, max=256, step=4),
            TuningConstraint(name="BLOCK_K", min=16, max=128, step=4),
            TuningConstraint(name="ELEMS_PER_THREAD", min=4, max=4, step=1),
        ]
        bench.tune_kernels(mfma_configs, tiling_constraints, ConvTuningSpec)
    else:
        if args.use_tuned:
            bench.load_tuned_results(args.use_tuned, ConvTuningSpec)

        if backend_name in ["iree", "wave"]:
            bench.compile_kernels()
            bench.benchmark_kernels()
        else:
            bench.benchmark_kernels_extern()

    # exec_args = [
    #     "iree-benchmark-module",
    #     f"--device={device}",
    #     "--device_allocator=caching",
    #     f"--module={vmfb_filename}",
    #     f"--function={entrypoint}",
    #     f"--benchmark_repetitions={args.iterations}",
    #     f"--input={image_shape}",
    #     f"--input={filter_shape}",
    # ]
