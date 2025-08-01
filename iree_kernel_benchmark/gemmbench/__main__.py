# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import asdict
import json
import math
import pandas as pd
import os
import optuna
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import argparse
import sys

from wave_lang.kernel.wave.constraints import MMAType
from ..utils import *
from .gemm_utils import *
from .iree_gemm import IREEGemmBenchmark
from .wave_gemm import WaveGemmBenchmark
from .torch_gemm import TorchGemmBenchmark
from .problems import get_gemm_configs, get_tk_gemm_configs, get_matching_configs

BACKEND_TO_GEMM_BENCH = {
    "torch": TorchGemmBenchmark,
    "wave": WaveGemmBenchmark,
    "iree": IREEGemmBenchmark,
}


def tune_gemm_kernels(
    configs: List[Tuple[str, GemmConfig]],
    kernel_dir: Path,
    vmfb_dir: Path,
    dump_dir: Path,
    mfma_configs: List[MMAType] = [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
        MMAType.F32_16x16x32_K8_F16,
        MMAType.F32_32x32x16_K8_F16,
    ],
    tiling_constraints: List[TuningConstraint] = [
        TuningConstraint(name="BLOCK_M", min=16, max=256, step=4),
        TuningConstraint(name="BLOCK_N", min=16, max=256, step=4),
        TuningConstraint(name="BLOCK_K", min=16, max=128, step=4),
    ],
    num_trials: int = 100,
):
    tuning_results = {}
    tuning_dir = Path("results/tuning/gemm")
    os.makedirs(tuning_dir, exist_ok=True)

    def tune_config(config: Tuple[str, GemmConfig]):
        tuning_result: tuple[
            float,
            GemmConfig,
            MMAType,
        ] = (
            math.inf,
            None,
            mfma_configs[0],
        )

        config_tag, kernel = config
        config_name = kernel.get_name()

        def objective(trial: optuna.Trial) -> float:
            nonlocal tuning_result

            block_sizes = [
                trial.suggest_int(
                    constraint.name,
                    constraint.min,
                    constraint.max,
                    step=constraint.step,
                )
                for constraint in tiling_constraints
            ]

            tuning_spec_params = {
                constraint.name: block_sizes[i]
                for i, constraint in enumerate(tiling_constraints)
            }

            tuning_spec = GemmTuningSpec(**tuning_spec_params)

            mfma_config = mfma_configs[
                trial.suggest_categorical("MFMA_INDEX", list(range(len(mfma_configs))))
            ]

            try:
                vmfb_dict = compile_gemm_kernels(
                    "wave",
                    [config],
                    kernel_dir,
                    vmfb_dir,
                    dump_dir,
                    None,
                    [],
                    {config_name: mfma_config},
                    {config_name: tuning_spec},
                )
            except Exception as e:
                print("Failed to compile, skipping", e)
                return math.inf

            output_csv_path = Path(f"temp.csv")

            try:
                result_df = benchmark_gemm_kernels(
                    "wave",
                    device,
                    vmfb_dict,
                    output_csv_path,
                    num_iterations=3,
                    debug=False,
                )
            except Exception as e:
                print("Failed runtime, skipping", e)
                return math.inf

            runtime = result_df["mean_microseconds"].mean()
            if runtime < tuning_result[0]:
                tuning_result = (runtime, tuning_spec, mfma_config)

            return runtime

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

        best_runtime, best_spec, best_mfma = tuning_result
        print("Optimal spec", asdict(best_spec))

        if best_spec and best_mfma:
            tuning_results[config_name] = {
                "block_sizes": asdict(best_spec),
                "mfma_variant": [best_mfma.name],
                "mean_microseconds": best_runtime,
            }
            with open(tuning_dir / "results.json", "w") as file:
                json.dump(tuning_results, file, indent=4)

    for config in configs:
        tune_config(config)


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
        "--target",
        help="The IREE hip target to compile for. The special value host_cpu results in a llvm-cpu benchmark instead of HIP, compiled for the host CPU.",
        type=str,
        default="gfx942",
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
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=[],
        help="List of data types to generate benchmarks for. Defaults to f16. Other options include (for example) f32, bf16, i8, f8E4M3FNUZ.",
    )
    parser.add_argument(
        "--raw_accumulators",
        action="store_true",
        help="If true, generate benchmark matmuls returning the raw accumulator type with no truncation. If false (default), generate benchmark matmuls where results are truncated and cast to the input element type.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[],
        help="List of matmul variants to filter benchmarks by. Default to all variants: NN, NT, TN, and TT.",
    )
    parser.add_argument(
        "--tag_regex",
        help="Regular expression for allowed benchmark tags. Defaults to all tags allowed.",
        default=".*",
    )
    parser.add_argument(
        "--config_regex",
        help="Regular expression for allowed benchmark configurations. Defaults to all allowed.",
        default=".*",
    )
    parser.add_argument(
        "--roofline",
        help="Comma separated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
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
    # Handle default values here, since list args are not compatible with defaulted lists.
    requested_dtypes = ["f16"] if not args.dtypes else list(args.dtypes)
    requested_variants = (
        ["NN", "NT", "TN", "TT"] if not args.variants else list(args.variants)
    )

    logging.basicConfig(level=args.log_level)

    if args.roofline:
        for dtype in requested_dtypes:
            roofline(
                args.roofline,
                f"{args.plot.split('.')[0]}_{dtype}.png",
                args.batch,
                dtype,
                args.model,
            )
        sys.exit()

    backend_name = args.backend

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "kernels"
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    kernel_dir.mkdir(parents=True, exist_ok=True)

    target = args.target
    extra_compiler_args = ["--" + x for x in list(args.Xiree_compile)]
    device = "local-task" if target == "host_cpu" else args.device

    configs: List[Tuple[str, GemmConfig]] = []
    for dtype in requested_dtypes:
        configs += get_gemm_configs(dtype, backend_name, args.raw_accumulators)
    configs = get_matching_configs(
        configs,
        requested_variants,
        args.tag_regex,
        args.config_regex,
    )

    bench_params = {
        "backend": backend_name,
        "kernel_type": "gemm",
        "device": device,
        "configs": configs,
        "kernel_dir": kernel_dir,
        "dump_dir": dump_dir,
        "debug": True,
        "target": target,
        "num_iterations": args.iterations,
    }

    bench: KernelBenchmark = BACKEND_TO_GEMM_BENCH[backend_name](**bench_params)
    bench.reduce_configs(args.max_kernels)
    print(f"Generated {len(bench.configs)} gemm configs.")

    if args.tune:
        # if args.tuning_config:
        #     with open(args.tuning_config, "r") as file:
        #         tuning_config = json.load(file)
        #     configs = [
        #         (tag, kernel)
        #         for tag, kernel in configs
        #         if kernel.get_name() in tuning_config["kernels"]
        #     ]
        #     num_trials = tuning_config["num_trials"]
        # else:
        #     num_trials = 100
        mfma_configs: List[MMAType] = [
            MMAType.F32_16x16x16_F16,
            MMAType.F32_32x32x8_F16,
            MMAType.F32_16x16x32_K8_F16,
            MMAType.F32_32x32x16_K8_F16,
        ]
        tiling_constraints: List[TuningConstraint] = [
            TuningConstraint(name="BLOCK_M", min=16, max=256, step=4),
            TuningConstraint(name="BLOCK_N", min=16, max=256, step=4),
            TuningConstraint(name="BLOCK_K", min=16, max=128, step=4),
        ]
        bench.tune_kernels(mfma_configs, tiling_constraints, GemmTuningSpec)
    else:
        if args.use_tuned:
            bench.load_tuned_results(args.use_tuned, GemmTuningSpec)

        if backend_name in ["iree", "wave"]:
            bench.compile_kernels()
            bench.benchmark_kernels()
        else:
            bench.benchmark_kernels_extern()
