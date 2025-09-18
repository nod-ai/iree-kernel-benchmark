import random
import time
from typing import override

from kernel_bench.utils.print_utils import get_logger

from .paradigm import TuningParadigm, TuningContext
from kernel_bench.utils.bench_utils import BenchmarkResult
from kernel_bench.utils.progress_context import MainProgress


class ParallelProgressTester(TuningParadigm):
    """Test implementation for demonstrating the new clean progress API."""

    def get_name(self) -> str:
        return "Progress Testing"

    @override
    def _tune(self, context: TuningContext, progress: MainProgress) -> BenchmarkResult:
        """Demonstrate the clean progress API with sub-progress bars."""

        # Configure main progress
        progress.configure(
            total=100, description="Multi-Stage Kernel Tuning", color="green"
        )

        # Stage 1: Compilation with sub-progress
        with progress.sub_progress("Compilation", 10, "yellow") as compile_progress:
            for i in range(10):
                compile_progress.update(i + 1)
                progress.update(i * 3, f"Compiling step {i+1}/10")
                time.sleep(random.uniform(0.1, 0.3))

        # # Stage 2: Optimization with sub-progress
        # with progress.sub_progress("Optimization", 15, "cyan") as opt_progress:
        #     for i in range(15):
        #         opt_progress.update(i + 1)
        #         progress.update(30 + i * 4, f"Optimizing trial {i+1}/15")
        #         time.sleep(random.uniform(0.1, 0.25))

        # # Stage 3: Validation with sub-progress
        # with progress.sub_progress("Validation", 5, "magenta") as val_progress:
        #     for i in range(5):
        #         val_progress.update(i + 1)
        #         progress.update(90 + i * 2, f"Validating config {i+1}/5")
        #         time.sleep(random.uniform(0.2, 0.4))

        # Final update
        progress.update(100, "Tuning Complete!")

        return context.bench.get_bench_result(0, False)
