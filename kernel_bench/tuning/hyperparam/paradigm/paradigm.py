from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
)

from kernel_bench.core.base import create_benchmark
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.core.utils import BenchmarkResult
from kernel_bench.utils.parallel import ProgressUpdate


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

        bench = context.bench
        context.bench = create_benchmark(
            bench.kernel_type, bench.backend, asdict(bench)
        )
        config = context.bench.config

        self.progress = ProgressUpdate(
            device_id=context.device_id,
            completed=0,
            total=context.num_trials,
            current=bench.config.get_name(),
            active=True,
            worker_id=context.worker_id,
        )
        self.progress_callback = progress_callback
        self._update_progress()

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

        self._update_progress(completed=self.progress.total, active=False)

        return TuningResult(
            name=config.get_name(),
            benchmark=best_result,
            improvement=improvement,
            speedup=speedup,
            hyperparams=best_result.tuning_config,
        )

    def _update_progress(
        self,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        active: Optional[bool] = None,
    ):
        if completed:
            self.progress.completed = completed
        if total:
            self.progress.total = total
        if active is not None:
            self.active = active
        self.progress_callback(self.progress)

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

            sat, violated = bench.tuning_spec.validate_constraints()
            if not sat:
                return bench.get_bench_result(math.inf, False)

        bench_result = bench.run_bench(
            f"hip://{context.device_id}", context.num_iterations
        )
        if not bench_result.ok:
            bench_result.mean_microseconds = math.inf
            bench_result.tflops = 0
        return bench_result
