from .utils import *
from .bayesian import *


def tune_kernels_parallel(
    configs: List[Tuple[str, OpConfig]],
    mfma_configs: List[Tuple[MMAType]],
    tiling_constraints: List[TuningConstraint],
    tuning_class: Type[TuningSpec],
    compile_kernel_func: Callable,
    bench_kernel_func: Callable,
    kernel_dir: os.PathLike,
    tuning_result_path: os.PathLike,
    num_iterations: int = 1,
    num_trials: int = 100,
    debug: bool = False,
    save_results: bool = True,
    tuning_paradigm: Optional[TuningParadigm] = None,
) -> Dict[str, Any]:
    """Backward compatible function for parallel kernel tuning."""
    if tuning_paradigm is None:
        tuning_paradigm = BayesianTuningParadigm()

    tuner = ParallelTuner(tuning_paradigm)
    return tuner.tune_kernels(
        configs=configs,
        mfma_configs=mfma_configs,
        tiling_constraints=tiling_constraints,
        tuning_class=tuning_class,
        compile_kernel_func=compile_kernel_func,
        bench_kernel_func=bench_kernel_func,
        kernel_dir=kernel_dir,
        tuning_result_path=tuning_result_path,
        num_iterations=num_iterations,
        num_trials=num_trials,
        debug=debug,
        save_results=save_results,
    )
