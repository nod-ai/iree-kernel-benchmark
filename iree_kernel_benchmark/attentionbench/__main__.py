import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from typing import Literal
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
import optuna
import argparse
import sys
from ..utils import *
from .attention_utils import *
from .attention_config import *
from .wave_attention import compile_attention_wave_vanilla
from .wave_bshd_attention import compile_attention_wave_bshd
from .iree_attention import compile_attention_iree
from .problems import get_attention_configs, get_attention_configs_gqa, ConfigList

type Backend = Literal["iree", "wave", "wavegqa"]


def compile_attention(
    tag: str,
    config: AttentionAttributes,
    kernel_dir: Path,
    vmfb_dir: Path,
    backend: Backend,
    extra_compiler_args: list[str] = [],
    dump_dir=None,
    mfma_variant: tuple[MMAType] = (
        MMAType.F32_32x32x16_K8_F16,
        MMAType.F32_32x32x8_F16,
    ),
    spec: TuningSpec = None,
):
    name = config.to_bmnk1k2().get_name()
    if dump_dir:
        dump_dir = Path(dump_dir)
        dpath = dump_dir / name
        extra_compiler_args.extend([f"--iree-hal-dump-executable-files-to={dpath}"])

    mlir_file = kernel_dir / f"{name}.mlir"
    vmfb_file = vmfb_dir / f"{name}.vmfb"

    if not mfma_variant:
        mfma_variant = (
            MMAType.F32_32x32x16_K8_F16,
            MMAType.F32_32x32x8_F16,
        )

    if not spec:
        if backend == "wavegqa":
            wg_tiles = [1, 1, 128, 128, 32]
        else:
            wg_tiles = [1, 128, 0, 0, 0]
        spec = TuningSpec(
            wg_tiles,
            [0, 0, 0, 0, 32],
            4,
            1,
            IntrinsicType.VMFMA_F32_32x32x16_F16,
            2,
            True,
        )

    if backend == "iree":
        mlir_file, vmfb_file = compile_attention_iree(
            config, spec, mlir_file, vmfb_file, dump_dir, extra_compiler_args
        )
    elif backend == "wave":
        mlir_file, vmfb_file = compile_attention_wave_vanilla(
            config, spec, mlir_file, vmfb_file, dump_dir, mfma_variant
        )
    elif backend == "wavegqa":
        mlir_file, vmfb_file = compile_attention_wave_bshd(
            config, spec, mlir_file, vmfb_file, dump_dir, mfma_variant
        )

    return (tag, config, mlir_file, vmfb_file)


def compile_attention_kernels(
    backend_name: str,
    configs: ConfigList,
    kernel_dir: Path,
    vmfb_dir: Path,
    dump_dir: Path,
    mfma_configs: dict[str, tuple[MMAType, MMAType]] = {},
    tuning_specs: dict[str, TuningSpec] = {},
) -> dict:
    vmfb_dict = {}

    def compile_args_generator():
        return itertools.starmap(
            lambda tag, config: (
                tag,
                config,
                kernel_dir,
                vmfb_dir,
                backend_name,
                [],
                dump_dir,
                (
                    mfma_configs.get(config.to_bshd().get_name())
                    if backend_name == "wavegqa"
                    else None
                ),
                (
                    tuning_specs.get(config.to_bshd().get_name())
                    if backend_name == "wavegqa"
                    else None
                ),
            ),
            configs,
        )

    if len(configs) < 5:
        compilation_results = [
            compile_attention(*args) for args in compile_args_generator()
        ]
    else:
        num_cpus = max(1, cpu_count() - 20)
        print(f"Using {num_cpus} CPUs for parallel processing.")
        manager = Manager()
        shared_vmfb_dict = manager.dict()

        with Pool(num_cpus) as pool:
            compilation_results = list(
                tqdm(
                    pool.istarmap(compile_attention, compile_args_generator()),
                    total=len(configs),
                    desc="Compiling Attention Kernels",
                )
            )
        vmfb_dict = shared_vmfb_dict

    error_count = 0
    for tag, config, mlir_file, vmfb_file in compilation_results:
        if vmfb_file:
            vmfb_dict[vmfb_file] = (tag, config)
        else:
            error_count += 1

    if len(configs) > 5:
        print(
            f"{len(configs) - error_count} Success, {error_count} Failed out of {len(configs)} configs"
        )
        print("Compilation process completed.")

    return dict(vmfb_dict)


def benchmark_attention_kernels(
    backend_name: str,
    vmfb_dict: dict,
    output_csv: Path,
    num_iterations: int = 3,
    debug=True,
) -> pd.DataFrame:
    results = []
    index = 0

    csv_dir = os.path.dirname(output_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    bench_items = vmfb_dict.items()
    if debug:
        bench_items = tqdm(vmfb_dict.items(), desc="Benchmarking Attention Kernels")

    for vmfb_filename, value in bench_items:
        tag = value[0]
        attn_attrs: AttentionAttributes = value[1]
        config: AttentionConfigBase = (
            attn_attrs.to_bshd()
            if backend_name == "wavegqa"
            else attn_attrs.to_bmnk1k2()
        )
        name = config.get_name()

        query_shape = config.get_query_shape()
        key_shape = config.get_key_shape()
        value_shape = config.get_value_shape()

        exec_args = [
            "iree-benchmark-module",
            f"--device={device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            "--function=main",
            f"--input={query_shape}",
            f"--input={key_shape}",
            f"--input={value_shape}",
            f"--benchmark_repetitions={num_iterations}",
        ]

        if backend_name.startswith("wave"):
            out_shape: str = config.get_output_shape()
            out_shape = "x".join(out_shape.split("x")[:-1] + ["f32"])
            exec_args.append(f"--input={out_shape}")
            exec_args += ["--function=isolated_benchmark"]
        elif backend_name == "iree":
            exec_args += ["--function=main"]

        # iree benchmark kernels
        ret_value, cmd_out, cmd_err = run_iree_command(exec_args)
        ok = ret_value == 0
        benchmark_gemm_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_gemm_mean_time_us = benchmark_gemm_mean_time_ms * 1000

        flops = config.get_flops()
        byte_count = config.get_byte_count()

        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_gemm_mean_time_us / 1e6)

        if backend_name == "wavegqa":
            config_bshd = attn_attrs.to_bshd()
            results.append(
                (
                    index,
                    tag,
                    name,
                    config_bshd.B,
                    config_bshd.H,
                    config_bshd.H_KV,
                    config_bshd.N_Q,
                    config_bshd.D_KV,
                    config_bshd.D_Q,
                    config_bshd.N_KV,
                    config.dtype,
                    round(benchmark_gemm_mean_time_us, 4),
                    round(arithmetic_intensity, 4),
                    round(tflops_per_second, 4),
                    ok,
                )
            )
        else:
            config_bmnk = attn_attrs.to_bmnk1k2()
            results.append(
                (
                    index,
                    tag,
                    name,
                    config_bmnk.B,
                    config_bmnk.M,
                    config_bmnk.N,
                    config_bmnk.K1,
                    config_bmnk.K2,
                    config_bmnk.dtype,
                    round(benchmark_gemm_mean_time_us, 4),
                    round(arithmetic_intensity, 4),
                    round(tflops_per_second, 4),
                    ok,
                )
            )
        index += 1

    if backend_name == "wavegqa":
        fieldnames = [
            "index",
            "tag",
            "name",
            "B",
            "H",
            "H_KV",
            "N_Q",
            "D_KV",
            "D_Q",
            "N_KV",
            "dtype",
            "mean_microseconds",
            "arithmetic_intensity",
            "tflops",
            "ok",
        ]
    else:
        fieldnames = [
            "index",
            "tag",
            "name",
            "B",
            "M",
            "N",
            "K1",
            "K2",
            "dtype",
            "mean_microseconds",
            "arithmetic_intensity",
            "tflops",
            "ok",
        ]

    write_results_to_csv(results, output_csv, fieldnames)
    if debug:
        print(f"Results written to {output_csv}")

    return pd.read_csv(output_csv)


def tune_attention_kernels(
    configs: ConfigList,
    kernel_dir: Path,
    vmfb_dir: Path,
    dump_dir: Path,
):
    mfma_configs = [
        (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ]
    tiling_constraints = [
        TuningConstraint(name="BLOCK_B", min=1, max=1, step=1),
        TuningConstraint(name="BLOCK_H", min=1, max=2, step=1),
        TuningConstraint(name="BLOCK_N_Q", min=16, max=128, step=16),
        TuningConstraint(name="BLOCK_D_KV", min=16, max=128, step=16),
        TuningConstraint(name="BLOCK_N_KV", min=16, max=64, step=16),
    ]

    tuning_results = {}
    tuning_dir = Path("results/tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    def tune_config(config):
        tuning_result: tuple[float, TuningSpec, tuple[MMAType]] = (
            math.inf,
            None,
            mfma_configs[0],
        )

        config_tag, kernel = config
        config_name = kernel.to_bshd().get_name()

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
            tuning_spec = TuningSpec(
                wg_tiles=block_sizes,
                reduction_tiles=[0 for _ in range(len(block_sizes))],
                M_warp=0,
                N_warp=0,
                intrinsic="none",
                waves_per_eu=2,
                denorm_flush=False,
            )

            mfma_config = mfma_configs[
                trial.suggest_categorical("MFMA_INDEX", list(range(len(mfma_configs))))
            ]

            try:
                vmfb_dict = compile_attention_kernels(
                    backend_name,
                    [config],
                    kernel_dir,
                    vmfb_dir,
                    dump_dir,
                    mfma_config,
                    {config_name: tuning_spec},
                )
            except:
                print("Failed to compile, skipping")
                return math.inf

            output_csv_path = Path(f"temp.csv")

            try:
                result_df = benchmark_attention_kernels(
                    backend_name,
                    vmfb_dict,
                    output_csv_path,
                    args.iterations,
                    debug=False,
                )
            except:
                print("Failed runtime, skipping")
                return math.inf

            runtime = result_df["mean_microseconds"].mean()
            if runtime < tuning_result[0]:
                tuning_result = (runtime, tuning_spec, mfma_config)

            return runtime

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, show_progress_bar=True)

        best_runtime, best_spec, best_mfma = tuning_result
        print("Optimal spec", best_spec)

        if best_spec and best_mfma:
            tuning_results[config_name] = {
                "spec": asdict(best_spec),
                "mfma": [mfma.name for mfma in best_mfma],
                "mean_microseconds": best_runtime,
            }
            with open(tuning_dir / "results.json", "w") as file:
                json.dump(tuning_results, file, indent=4)

    for config in configs:
        tune_config(config)


@dataclass
class TuningConstraint:
    name: str
    min: int
    max: int
    step: int
    exponential: bool = False

    def get_range(self) -> list[int]:
        range = []

        curr = self.min
        while curr <= self.max:
            range.append(curr)
            if self.exponential:
                curr *= self.step
            else:
                if curr == 1 and self.step > 1:
                    curr += self.step - 1
                else:
                    curr += self.step

        return range


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
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument(
        "--batch", help="roofline on certain batch", type=int, default=None
    )
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument(
        "--backend",
        choices=["iree", "wave", "wavegqa"],
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
        "--use_tuned",
        type=str,
        default=None,
        help="Path to json file with tuned results.",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Title of run for save path"
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.roofline:
        roofline(args.roofline, args.plot, args.batch, args.dtype, args.model)
        sys.exit()

    backend_name = args.backend

    mfma_config = (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x16_K8_F16)

    if backend_name == "wavegqa":
        configs = get_attention_configs_gqa()
    elif backend_name == "iree":
        configs = get_attention_configs(use_fp8=True)
    else:
        configs = get_attention_configs(use_fp8=False)
    print(f"Generated {len(configs)} attention configs.")

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "attention" / backend_name / "mlir"
    vmfb_dir = repo_root / "attention" / backend_name / "vmfb"
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    device = args.device
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)

    if args.tune:
        tune_attention_kernels(configs, kernel_dir, vmfb_dir, dump_dir)

    else:
        specs: dict[str, TuningSpec] = {}
        mfma_configs: dict[str, tuple[MMAType, MMAType]] = {}

        def to_mma(str_config: list[str]) -> tuple[MMAType, MMAType]:
            return (
                MMAType[str_config[0]],
                MMAType[str_config[1]],
            )

        if args.use_tuned:
            with open(args.use_tuned, "r") as file:
                tuned_data: dict[str, dict] = json.load(file)
                specs = {
                    kernel_name: TuningSpec(**tune_result["spec"])
                    for kernel_name, tune_result in tuned_data.items()
                }
                mfma_configs = {
                    kernel_name: to_mma(tune_result["mfma"])
                    for kernel_name, tune_result in tuned_data.items()
                }

        vmfb_dict = compile_attention_kernels(
            backend_name,
            configs,
            kernel_dir,
            vmfb_dir,
            dump_dir,
            mfma_configs,
            tuning_specs=specs,
        )

        output_csv_base = f"attention_{backend_name}" + (
            f"_{args.title}" if args.title else ""
        )
        output_csv_path = Path(f"results/attention/{output_csv_base}.csv")
        print(f"Results will be written to {output_csv_path}")

        benchmark_attention_kernels(
            backend_name, vmfb_dict, output_csv_path, args.iterations
        )
