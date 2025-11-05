import argparse
from kernel_bench.core.template import batch_benchmark, batch_compile_iree_benches
from kernel_bench.kernels.gemm.backends.wave_gemm import WaveGemmBenchmark
from kernel_bench.kernels.gemm.problems import get_meta_gemms
from kernel_bench.utils.bench_utils import write_results_to_csv, write_results_to_json
from kernel_bench.utils.paths import PathConfig
from kernel_bench.utils.plot_utils import (
    create_comparison_plot,
    load_all_backend_results,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType

options = {
    "wave_nt": {
        "schedule": SchedulingType.NONE,
        "use_global_to_shared": False,
    },
    "wave_nt_g2s": {
        "schedule": SchedulingType.NONE,
        "use_global_to_shared": True,
    },
    "wave_nt_double_buffer": {
        "schedule": SchedulingType.NONE,
        "use_global_to_shared": False,
        "multi_buffer_count": 2,
    },
    "wave_nt_g2s_double_buffer": {
        "schedule": SchedulingType.NONE,
        "use_global_to_shared": True,
        "multi_buffer_count": 2,
    },
    "wave_nt_prefetch": {
        "schedule": SchedulingType.PREFETCH,
        "use_global_to_shared": False,
    },
    "wave_nt_prefetch_g2s": {
        "schedule": SchedulingType.PREFETCH,
        "use_global_to_shared": True,
    },
    "wave_nt_prefetch_double_buffer": {
        "schedule": SchedulingType.PREFETCH,
        "use_global_to_shared": False,
        "multi_buffer_count": 2,
    },
    "wave_nt_prefetch_g2s_double_buffer": {
        "schedule": SchedulingType.PREFETCH,
        "use_global_to_shared": True,
        "multi_buffer_count": 2,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--plot", required=False, type=str, help="Path to save comparison plot"
    )
    parser.add_argument(
        "--use_tag",
        action="store_true",
        help="Use tag names on x-axis instead of shape configurations",
    )
    args = parser.parse_args()

    meta_gemms = [
        (tag, config) for tag, config in get_meta_gemms() if tag == "meta-4-shapes"
    ]
    path_config = PathConfig.default()

    bench_mappings = []

    for title, compile_options in options.items():
        benches = []
        for tag, problem in meta_gemms:
            wave_bench = WaveGemmBenchmark(
                tag, "wave", "gemm", "mi350x", problem, path_config, title
            )
            wave_bench.replace_compile_options(**compile_options)
            benches.append(wave_bench)
        bench_mappings.append((title, benches))

    all_benches = []
    for title, benches in bench_mappings:
        all_benches.extend(benches)
    all_bench_results = batch_benchmark(
        all_benches, "hip", validate_numerics=False, unique_ids=True
    )

    for i, title in enumerate(options.keys()):
        start_idx = i * len(meta_gemms)
        bench_results = all_bench_results[start_idx : start_idx + len(meta_gemms)]

        output_base = f"gemm_{title}"
        output_csv_dir = path_config.csv_for("gemm")
        output_json_dir = path_config.json_for("gemm")
        output_csv_path = output_csv_dir / f"{output_base}.csv"
        output_json_path = output_json_dir / f"{output_base}.json"

        write_results_to_csv(bench_results, output_csv_path, title)
        print(f"Results written to {output_csv_path}")
        write_results_to_json(bench_results, output_json_path, title)
        print(f"Results written to {output_json_path}")

    if args.plot:
        backend_names = list(options.keys())
        backend_results = load_all_backend_results(
            backend_names, "gemm", "wave", "mi350x"
        )
        create_comparison_plot(
            backend_results=backend_results,
            kernel_type="gemm",
            machine="mi350x",
            backend_names=backend_names,
            save_path=args.plot,
            use_tag=args.use_tag,
        )
