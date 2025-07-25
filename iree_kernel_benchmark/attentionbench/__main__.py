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
from iree_kernel_benchmark.gemmbench.gemm_utils import GemmConfig
from wave_lang.kernel.wave.constraints import MMAType
from ..utils import *
from .attention_utils import *
from .attention_config import *
from .wave_attention import compile_attention_wave_vanilla, compile_attention_wave_bshd
from .iree_attention import compile_attention_iree
from .torch_attention import benchmark_torch_attention
from .problems import get_attention_configs, get_attention_configs_gqa, ConfigList

type Backend = Literal["iree", "wave", "wavegqa"]
type AttentionTuningSpec = AttentionBMNKTuningSpec | AttentionBSHDTuningSpec | IREEAttentionTuningSpec


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
    spec: Optional[AttentionTuningSpec] = None,
):
    name = config.get_name()
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

    if backend == "iree":
        mlir_file, vmfb_file = compile_attention_iree(
            config, mlir_file, vmfb_file, spec, dump_dir, extra_compiler_args
        )
    elif backend == "wave":
        mlir_file, vmfb_file = compile_attention_wave_vanilla(
            config, mlir_file, vmfb_file, spec, dump_dir, mfma_variant
        )
    elif backend == "wavegqa":
        mlir_file, vmfb_file = compile_attention_wave_bshd(
            config, mlir_file, vmfb_file, spec, dump_dir, mfma_variant
        )

    return (tag, config, mlir_file, vmfb_file)


def compile_attention_kernels(
    backend_name: str,
    configs: ConfigList,
    kernel_dir: Path,
    vmfb_dir: Path,
    dump_dir: Path,
    mfma_configs: dict[str, tuple[MMAType, MMAType]] = {},
    tuning_specs: dict[str, AttentionTuningSpec] = {},
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
                mfma_configs.get(config.get_name()),
                tuning_specs.get(config.get_name()),
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


def save_results(
    configs: ConfigList, runtimes_us: List[float], ok: List[bool], output_csv: Path
) -> pd.DataFrame:

    csv_dir = os.path.dirname(output_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    index = 0
    results = []

    for i, (tag, attn_attrs) in enumerate(configs):
        benchmark_mean_time_us = runtimes_us[i]

        config: AttentionConfigBase = (
            attn_attrs.to_bshd()
            if backend_name == "wavegqa"
            else attn_attrs.to_bmnk1k2()
        )

        flops = config.get_flops()
        byte_count = config.get_byte_count()

        arithmetic_intensity = flops / byte_count
        if benchmark_mean_time_us == 0:
            tflops_per_second = 0
        else:
            tflops_per_second = (flops / 1e12) / (benchmark_mean_time_us / 1e6)

        if backend_name == "wavegqa":
            config_bshd = attn_attrs.to_bshd()
            results.append(
                (
                    index,
                    tag,
                    config.get_name(),
                    config_bshd.B,
                    config_bshd.H,
                    config_bshd.H_KV,
                    config_bshd.N_Q,
                    config_bshd.D_KV,
                    config_bshd.D_Q,
                    config_bshd.N_KV,
                    config.dtype,
                    round(benchmark_mean_time_us, 4),
                    round(arithmetic_intensity, 4),
                    round(tflops_per_second, 4),
                    ok[i],
                )
            )
        else:
            config_bmnk = attn_attrs.to_bmnk1k2()
            results.append(
                (
                    index,
                    tag,
                    config.get_name(),
                    config_bmnk.B,
                    config_bmnk.M,
                    config_bmnk.N,
                    config_bmnk.K1,
                    config_bmnk.K2,
                    config_bmnk.dtype,
                    round(benchmark_mean_time_us, 4),
                    round(arithmetic_intensity, 4),
                    round(tflops_per_second, 4),
                    ok[i],
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
    print(f"Results written to {output_csv}")

    return pd.read_csv(output_csv)


def benchmark_attention_kernels(
    backend_name: str,
    device: str,
    vmfb_dict: dict,
    output_csv: Path,
    num_iterations: int = 3,
    debug=True,
) -> pd.DataFrame:

    bench_items = vmfb_dict.items()
    if debug:
        bench_items = tqdm(vmfb_dict.items(), desc="Benchmarking Attention Kernels")

    runtimes = []
    statuses = []
    configs = []

    for vmfb_filename, value in bench_items:
        tag = value[0]
        attn_attrs: AttentionAttributes = value[1]
        config: AttentionConfigBase = (
            attn_attrs.to_bshd()
            if backend_name == "wavegqa"
            else attn_attrs.to_bmnk1k2()
        )

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

        runtimes.append(benchmark_gemm_mean_time_us)
        statuses.append(ok)
        configs.append((tag, attn_attrs))

    save_results(configs, runtimes, statuses, output_csv)


def benchmark_extern_attention_kernels(
    backend: str, configs: ConfigList, output_csv: Path, num_iterations: int = 3
):
    runtimes_us = []
    statuses = []

    for tag, config in tqdm(configs, f"Benchmarking {backend} attention kernels"):
        if backend == "torch":
            mean_us = benchmark_torch_attention(config, num_iterations)
            if mean_us:
                runtimes_us.append(mean_us)
                statuses.append(True)
            else:
                runtimes_us.append(0)
                statuses.append(False)

    return save_results(configs, runtimes_us, statuses, output_csv)


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


def tune_attention_kernels(
    configs: ConfigList,
    kernel_dir: Path,
    vmfb_dir: Path,
    dump_dir: Path,
    mfma_configs: List[Tuple[MMAType, MMAType]] = [
        (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
    tiling_constraints: List[TuningConstraint] = [
        TuningConstraint(name="BLOCK_B", min=1, max=1, step=1),
        TuningConstraint(name="BLOCK_H", min=1, max=2, step=1),
        TuningConstraint(name="BLOCK_N_Q", min=16, max=128, step=16),
        TuningConstraint(name="BLOCK_D_KV", min=16, max=128, step=16),
        TuningConstraint(name="BLOCK_N_KV", min=16, max=64, step=16),
    ],
):
    tuning_results = {}
    tuning_dir = Path("results/tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    def tune_config(config: Tuple[str, AttentionAttributes]):
        tuning_result: tuple[
            float,
            AttentionBMNKTuningSpec | AttentionBSHDTuningSpec,
            tuple[MMAType, MMAType],
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

            if kernel.attention_type == "bshd":
                tuning_spec = AttentionBSHDTuningSpec(**tuning_spec_params)
            else:
                tuning_spec = AttentionBMNKTuningSpec(**tuning_spec_params)

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
        print("Optimal spec", asdict(best_spec))

        if best_spec and best_mfma:
            tuning_results[config_name] = {
                "block_sizes": asdict(best_spec),
                "mfma_variant": [mfma.name for mfma in best_mfma],
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

    configs: List[Tuple[str, AttentionAttributes]] = reduce_configs(
        configs, args.max_kernels
    )
    print(f"Generated {len(configs)} attention configs.")

    output_csv_base = f"attention_{backend_name}" + (
        f"_{args.title}" if args.title else ""
    )
    output_csv_path = Path(f"results/attention/{output_csv_base}.csv")

    if backend_name not in ["iree", "wave", "wavegqa"]:
        benchmark_extern_attention_kernels(backend_name, configs, output_csv_path)
        exit(0)

    attention_type = configs[0][1].attention_type

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
        specs: dict[str, AttentionTuningSpec] = {}
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
                    kernel_name: (
                        AttentionBMNKTuningSpec(**tune_result["block_sizes"])
                        if attention_type == "bmnk"
                        else AttentionBSHDTuningSpec(**tune_result["block_sizes"])
                    )
                    for kernel_name, tune_result in tuned_data.items()
                }
                mfma_configs = {
                    kernel_name: to_mma(tune_result["mfma_variant"])
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

        benchmark_attention_kernels(
            backend_name, device, vmfb_dict, output_csv_path, args.iterations
        )
