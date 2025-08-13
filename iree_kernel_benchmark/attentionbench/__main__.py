from multiprocessing import Pool, cpu_count, Manager
import logging
from typing import Literal
from pathlib import Path
import json
import argparse
import sys
from wave_lang.kernel.wave.constraints import MMAType
from ..utils import *
from .attention_utils import *
from .attention_config import *
from .wave_attention import WaveAttentionGQABenchmark, WaveAttentionMHABenchmark
from .iree_attention import IREEAttentionBenchmark
from .torch_attention import TorchAttentionBenchmark
from .problems import get_attention_configs, get_attention_configs_gqa

type Backend = Literal["iree", "wave", "wavegqa"]
type AttentionTuningSpec = AttentionBMNKTuningSpec | AttentionBSHDTuningSpec

BACKEND_TO_ATTENTION_BENCH = {
    "torch": TorchAttentionBenchmark,
    "wave": WaveAttentionMHABenchmark,
    "wavegqa": WaveAttentionGQABenchmark,
    "iree": IREEAttentionBenchmark,
}

BACKEND_TO_ATTENTION_TUNING_SPEC = {
    "torch": AttentionBMNKTuningSpec,
    "wave": AttentionBMNKTuningSpec,
    "wavegqa": AttentionBSHDTuningSpec,
    "iree": AttentionBMNKTuningSpec,
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
        "--roofline",
        help="Comma seperated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument(
        "--device",
        help="The IREE device to execute benchmarks on",
        type=str,
        default="hip",
    )
    parser.add_argument(
        "--target",
        help="The IREE hip target to compile for.",
        type=str,
        default="gfx942",
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

    if args.roofline:
        roofline(args.roofline, args.plot, args.batch, args.dtype, args.model)
        sys.exit()

    backend_name = args.backend

    mfma_config = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x16_K8_F16)

    configs = []
    if args.load_problems:
        config_class = (
            AttentionConfigBSHD if backend_name == "wavegqa" else AttentionConfigBMNK
        )
        configs = load_configs(
            args.load_problems,
            "attention",
            backend_name,
            config_class,
        )

    if len(configs) == 0:
        if backend_name == "wavegqa":
            configs = [
                (tag, config.to_bshd()) for tag, config in get_attention_configs_gqa()
            ]
        elif backend_name == "iree":
            configs = [
                (tag, config.to_bmnk1k2())
                for tag, config in get_attention_configs(use_fp8=True)
            ]
        else:
            configs = [
                (tag, config.to_bmnk1k2())
                for tag, config in get_attention_configs(use_fp8=False)
            ]

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "kernels"
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    device = args.device
    kernel_dir.mkdir(parents=True, exist_ok=True)

    bench_params = {
        "backend": backend_name,
        "kernel_type": "attention",
        "device": device,
        "target": args.target,
        "configs": configs,
        "kernel_dir": kernel_dir,
        "dump_dir": dump_dir,
        "debug": True,
        "num_iterations": args.iterations,
    }

    bench: KernelBenchmark = BACKEND_TO_ATTENTION_BENCH[backend_name](**bench_params)
    bench.reduce_configs(args.max_kernels)
    print(f"Generated {len(bench.configs)} attention configs.")

    tuning_spec_class = BACKEND_TO_ATTENTION_TUNING_SPEC[backend_name]

    if args.tune:
        if backend_name == "wavegqa":
            mfma_configs: List[Tuple[MMAType, MMAType]] = [
                (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
            ]
            tiling_constraints: List[TuningConstraint] = [
                TuningConstraint(name="BLOCK_B", min=1, max=1, step=1),
                TuningConstraint(name="BLOCK_H", min=1, max=2, step=1),
                TuningConstraint(name="BLOCK_N_Q", min=16, max=128, step=16),
                TuningConstraint(name="BLOCK_D_KV", min=16, max=128, step=16),
                TuningConstraint(name="BLOCK_N_KV", min=16, max=64, step=16),
            ]
        else:
            mfma_configs: List[Tuple[MMAType, MMAType]] = [
                (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
                (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
            ]
            tiling_constraints: List[TuningConstraint] = [
                TuningConstraint(name="BLOCK_B", min=1, max=1, step=1),
                TuningConstraint(name="BLOCK_M", min=32, max=256, step=8),
                TuningConstraint(name="BLOCK_N", min=16, max=128, step=4),
                TuningConstraint(name="BLOCK_K2", min=32, max=256, step=8),
            ]

        bench.tune_kernels(
            mfma_configs,
            tiling_constraints,
            tuning_spec_class,
            num_trials=args.num_trials,
        )

    else:
        if args.use_tuned:
            bench.load_tuned_results(args.use_tuned, tuning_spec_class)

        if backend_name in ["iree", "wave", "wavegqa"]:
            bench.compile_kernels()
            bench.benchmark_kernels()
        else:
            bench.benchmark_kernels_extern()
