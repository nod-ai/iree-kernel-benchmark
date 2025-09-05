from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
)

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.core.utils import BenchmarkResult


@dataclass
class TuningContext:
    """Context object containing all necessary information for tuning."""

    bench: KernelBenchmark
    device_id: int
    num_iterations: int
    num_trials: int
    debug: bool = False
    worker_id: int = 0


@dataclass
class TuningResult:
    """Result of a tuning run."""

    name: str
    benchmark: BenchmarkResult
    improvement: bool
    speedup: float
    hyperparams: Optional[Dict[str, Any]] = None


class TuningParadigm(ABC):
    """Abstract base class for different tuning paradigms."""

    def tune(self, context: TuningContext, progress_callback: Callable) -> TuningResult:
        """Run the tuning process and return the best result."""
        config = context.bench.config

        base_result = self._benchmark(context)
        tuned_result = self._tune(context, progress_callback)

        base_runtime = base_result.mean_microseconds
        tuned_runtime = tuned_result.mean_microseconds

        if not tuned_result.ok or base_runtime < tuned_runtime:
            best_result = base_result
            improvement = False
            speedup = 0
        else:
            best_result = tuned_result
            improvement = True
            speedup = base_runtime / tuned_runtime

        return TuningResult(
            name=config.get_name(),
            benchmark=best_result,
            improvement=improvement,
            speedup=speedup,
            hyperparams=best_result.tuning_config,
        )

    @abstractmethod
    def _tune(
        self, context: TuningContext, progress_callback: Callable
    ) -> BenchmarkResult:
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this tuning paradigm."""
        pass

    def _benchmark(
        self,
        context: TuningContext,
        param_values: Optional[Dict[str, int]] = None,
    ) -> BenchmarkResult:
        """Compile and benchmark a kernel configuration."""
        bench = context.bench

        bench.tuning_spec.clear()
        if param_values:
            for name, val in param_values.items():
                bench.tuning_spec.set_parameter(name, val)

        bench_result = bench.run_bench(
            f"hip://{context.device_id}", context.num_iterations
        )
        return bench_result
