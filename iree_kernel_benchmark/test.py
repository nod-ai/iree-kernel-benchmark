from .gemmbench.problems import get_b200_gemm_configs_time
from .utils import *

configs = get_b200_gemm_configs_time("b200")

results = []

for i, (tag, runtime_us, config) in enumerate(configs):
    ok = runtime_us > 0
    arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(config, runtime_us)

    result = BenchmarkResult(
        index=i,
        machine="MI355X",
        kernel_type="gemm",
        backend="B200",
        tag=tag,
        name=config.get_name(),
        dims=config.get_dim_names(),
        shape=config.to_dict(),
        problem=asdict(config),
        tuning_config=None,
        mean_microseconds=round(runtime_us, 4),
        arithmetic_intensity=round(arithmetic_intensity, 4),
        tflops=round(tflops_per_second, 4),
        ok=ok,
    )
    results.append(result)

write_results_to_json(results, "results/json/gemm/gemm_b200.json")
