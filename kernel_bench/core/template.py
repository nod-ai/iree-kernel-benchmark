from abc import ABC, abstractmethod
import math
from multiprocessing import Pool, cpu_count
import os
import signal
import traceback
from uuid import uuid4
from sympy import Symbol
from dataclasses import dataclass, field
from dataclass_wizard import asdict
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm
from kernel_bench.utils.dtypes.device_context import DeviceContext
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.wave import LaunchableWave

from kernel_bench.utils.parallel_utils.isolated_runtime import (
    isolated_validate_numerics,
)
from kernel_bench.utils.print_utils import get_logger
from kernel_bench.utils.paths import PathConfig

from kernel_bench.config.base import OpConfig
from ..utils.bench_utils import (
    BenchmarkResult,
    get_kernel_perf_stats,
    redirect_stderr_to_file,
)
from kernel_bench.utils.iree_utils import bench_kernel_ireert
from kernel_bench.tuning.hyperparam.parameters import (
    TuningParameter,
    TuningSpec,
    ParameterSymbol,
)


class CompilationTimeoutError(Exception):
    """Exception raised when compilation exceeds timeout"""

    pass


class KernelValidationError(Exception):
    """Exception raised when kernel configuration is invalid"""

    pass


class TimeoutContext:
    """Context manager for setting a timeout on function execution"""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.old_handler = None

    def timeout_handler(self, signum, frame):
        raise CompilationTimeoutError(
            f"Compilation timed out after {self.timeout_seconds} seconds"
        )

    def __enter__(self):
        # Set up the signal handler
        self.old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.timeout_seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)


@dataclass
class KernelBenchmark(ABC):
    tag: str
    backend: str
    kernel_type: str
    machine: str
    config: OpConfig
    path_config: PathConfig

    def __post_init__(self):
        if not self.validate_config():
            raise KernelValidationError(
                f"Config {self.config.get_name()} invalid for {self.kernel_type} on backend {self.backend}"
            )

        self._tuning_spec = TuningSpec()
        self._param_symbols = {}

        self.device_ctx = DeviceContext.from_machine(self.machine)

        self.logger = get_logger()
        self.setup_parameters()

    def validate_config(self) -> bool:
        """Override to validate problem for given configuration"""
        return True

    def add_params(self, params: List[TuningParameter]) -> List[ParameterSymbol]:
        """Register multiple parameters and return their corresponding symbols"""
        symbols = []

        for param in params:
            self._tuning_spec.add_parameter(param.name, param)

            param_symbol = ParameterSymbol(param.name, self._tuning_spec)
            self._param_symbols[param.name] = param_symbol
            symbols.append(param_symbol)

        return symbols

    def add_param(self, name: str, bounds, **kwargs) -> ParameterSymbol:
        """Register and return a parameter symbol"""
        param = TuningParameter(name, bounds, **kwargs)
        self._tuning_spec.add_parameter(name, param)

        param_symbol = ParameterSymbol(name, self._tuning_spec)
        self._param_symbols[name] = param_symbol
        return param_symbol

    def add_constraint(self, constraint_expr, name: str = None):
        """Add constraint using SymPy expression or string"""
        self._tuning_spec.add_constraint(constraint_expr, name)

    def get_param(self, name: str):
        """Get parameter value by name"""
        return self._tuning_spec.get_parameter_value(name)

    def setup_parameters(self):
        """Override to define parameters and constraints"""
        pass

    def validate_numerics(self, device: str) -> bool:
        """Validate numerical accuracy of kernel before benchmarking"""
        return True

    def get_bench_result(self, runtime_us: float, ok: bool):
        arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
            self.config, runtime_us if ok else math.inf
        )

        tuning_config = self._tuning_spec.to_dict() or None

        return BenchmarkResult(
            machine=self.device_ctx.machine,
            kernel_type=self.kernel_type,
            backend=self.backend,
            tag=self.tag,
            name=self.config.get_name(),
            dims=self.config.get_dim_names(),
            shape=self.config.to_dict(),
            problem=asdict(self.config),
            tuning_config=tuning_config,
            mean_microseconds=round(runtime_us, 4),
            arithmetic_intensity=round(arithmetic_intensity, 4),
            tflops=round(tflops_per_second, 4),
            ok=ok,
        )

    @property
    def tuning_spec(self):
        return self._tuning_spec

    def validate_constraints(
        self, param_values: dict[str, Any] = None
    ) -> tuple[bool, dict[str, float]]:
        return self._tuning_spec.validate_constraints(param_values)

    def load_tuning_spec(self, new_spec: TuningSpec):
        self._tuning_spec = new_spec

    def load_tuned_config(self, obj: dict[str, Any]):
        self.tuning_spec.load_from_dict(obj)

    def update_parameter_values(self, param_values: dict[str, int]):
        for name, val in param_values.items():
            self.tuning_spec.set_parameter(name, val)

    @abstractmethod
    def run_bench(
        self, device: str, num_iterations: int = 1, timeout: Optional[float] = None
    ) -> BenchmarkResult:
        pass


@dataclass
class IREEKernelBenchmark(KernelBenchmark):
    @abstractmethod
    def compile_to_vmfb(self, mlir_path: Path, vmfb_path: Path) -> bool:
        pass

    @abstractmethod
    def get_runtime_args(self) -> List[str]:
        pass

    def bench_vmfb(
        self,
        vmfb_filename: PathLike,
        device: str,
        num_iterations: int = 3,
        timeout: Optional[float] = None,
    ) -> BenchmarkResult:
        runtime_us, ok = bench_kernel_ireert(
            vmfb_filename,
            self.get_runtime_args(),
            num_iterations=1,
            device=device,
            timeout=timeout,
        )
        return self.get_bench_result(runtime_us, ok)

    def run_bench(self, device, num_iterations=1, timeout=None):
        mlir_dir = self.path_config.mlir_for(self.kernel_type, self.backend)
        vmfb_dir = self.path_config.vmfb_for(self.kernel_type, self.backend)

        mlir_path = mlir_dir / f"{self.config.get_name()}.mlir"
        vmfb_path = vmfb_dir / f"{self.config.get_name()}.vmfb"

        compile_success = self.compile_to_vmfb(mlir_path, vmfb_path)
        if not compile_success:
            return self.get_bench_result(0, False)

        return self.bench_vmfb(vmfb_path, device, num_iterations, timeout)


@dataclass
class WaveTemplate:
    launchable: LaunchableWave
    hyperparams: Dict[Symbol, Any]
    dynamic_symbols: List[Symbol] = field(default_factory=list)


@dataclass
class WaveKernelBenchmark(IREEKernelBenchmark):
    @abstractmethod
    def load_wave_kernel(self) -> WaveTemplate:
        pass

    @abstractmethod
    def extra_compile_options(self) -> WaveCompileOptions:
        pass

    def get_compile_options(
        self, kernel: WaveTemplate, vmfb_path: Optional[Path] = None
    ) -> WaveCompileOptions:
        compile_options = self.extra_compile_options()

        if vmfb_path:
            compile_options.create_vmfb_file = vmfb_path
        compile_options.subs = kernel.hyperparams
        compile_options.dynamic_symbols = kernel.dynamic_symbols
        compile_options.iree_launch_async = False
        compile_options.run_bench = False
        compile_options.device = "hip"
        compile_options.target = self.device_ctx.hip_target
        compile_options.dump_intermediates = (
            self.path_config.dump_dir_for("wave") / self.config.get_name()
        )

        return compile_options

    def compile_to_vmfb(self, mlir_path, vmfb_path):
        try:
            kernel = self.load_wave_kernel()
            compile_options = self.get_compile_options(kernel, vmfb_path)

            # dump_file = self.path_config.dump_for("wave", self.config.get_name())
            dump_file = None
            if dump_file:
                with redirect_stderr_to_file(dump_file):
                    compile_options.mlir_print_ir_after_all = True
                    result = wave_compile(compile_options, kernel.launchable)
            else:
                result = wave_compile(compile_options, kernel.launchable)

            with open(mlir_path, "w") as mlir_out:
                mlir_out.write(result.asm)

            return True

        except Exception as e:
            self.logger.error(f"Failed to compile {self.config.get_name()}: {e}")
            self.logger.error(traceback.format_exc())
            return False


type CompileResult = Tuple[OpConfig, Optional[Path], bool]


def compile_iree_bench(bench: IREEKernelBenchmark, kernel_name: str) -> CompileResult:
    try:
        mlir_dir = bench.path_config.mlir_for(bench.kernel_type, bench.backend)
        vmfb_dir = bench.path_config.vmfb_for(bench.kernel_type, bench.backend)

        os.makedirs(mlir_dir, exist_ok=True)
        os.makedirs(vmfb_dir, exist_ok=True)

        mlir_path = mlir_dir / f"{kernel_name}.mlir"
        vmfb_path = vmfb_dir / f"{kernel_name}.vmfb"

        with TimeoutContext(60):
            success = bench.compile_to_vmfb(mlir_path, vmfb_path)

        return bench.config, vmfb_path, success

    except CompilationTimeoutError as e:
        get_logger().error(f"Compilation timed out for {bench.config.get_name()}: {e}")
        return bench.config, None, False
    except Exception as e:
        get_logger().error(f"Compilation failed for {bench.config.get_name()}: {e}")
        return bench.config, None, False


def batch_compile_iree_benches(
    iree_benches: List[IREEKernelBenchmark],
    callback: Optional[Callable[[CompileResult], None]] = None,
    verbose=False,
    unique_id=False,
) -> List[CompileResult]:
    """
    Compile a list of IREE-based kernel benchmarks. Compilation results
    guaranteed to preserve initial input order.

    Returns: CompileResult = Tuple[OpConfig, Optional[Path], bool]
    """

    logger = get_logger()

    def tag_name(name: str):
        if not unique_id:
            return name
        id_suffix = str(uuid4()).replace("-", "_")
        return f"{name}_{id_suffix}"

    compile_args = [
        (bench, tag_name(bench.config.get_name())) for bench in iree_benches
    ]

    num_cpus = max(1, cpu_count() - 20)
    num_cpus = min(num_cpus, len(iree_benches))
    if verbose:
        logger.info(f"Using {num_cpus} CPUs for parallel compilation.")

    with Pool(num_cpus) as pool:
        compilation_iterator = pool.istarmap(compile_iree_bench, compile_args)
        compilation_results = []

        for result in tqdm(
            compilation_iterator,
            total=len(iree_benches),
            desc=f"Compiling {iree_benches[0].kernel_type.capitalize()} Kernels",
            disable=callback is not None,
        ):
            compilation_results.append(result)
            if callback:
                callback(result)

    assert len(iree_benches) == len(compilation_results)

    success_count = sum([success for _, _, success in compilation_results])
    error_count = len(iree_benches) - success_count

    if verbose:
        (logger.info if error_count == 0 else logger.error)(
            f"{success_count} Success, {error_count} Failed out of {len(compilation_results)} configs"
        )
        logger.info("Compilation process completed.")

    return compilation_results


def batch_validate(
    benches: List[KernelBenchmark],
    device: str,
    verbose=False,
) -> List[bool]:
    logger = get_logger()

    results = []
    for bench in tqdm(benches, desc="Validating kernel numerics"):
        validation_result, error_msg = isolated_validate_numerics(bench, device)
        if error_msg:
            logger.error(
                f"Validation subprocess error for {bench.config.get_name()}: {error_msg}"
            )
        results.append(validation_result)

    if verbose:
        success_count = sum(results)
        error_count = len(results) - success_count
        (logger.info if error_count == 0 else logger.error)(
            f"Numerical check: {success_count} Success, {error_count} Failed out of {len(results)} configs"
        )
        logger.info("Validation process completed.")

    return results


def batch_benchmark(
    benches: List[KernelBenchmark],
    device: str,
    num_iterations: int = 1,
    timeout: Optional[float] = None,
    compile_callback: Optional[Callable[[CompileResult], None]] = None,
    bench_callback: Optional[Callable[[BenchmarkResult], None]] = None,
    validate_numerics=True,
    verbose=False,
    unique_ids=False,
) -> List[BenchmarkResult]:
    """
    Benchmark a list of kernel benchmarks.

    First compiles all IREE-based benches in batch, then benchmarks all benches
    in order while preserving the original input order.
    """

    # Compile all IREE-based benches
    iree_benches = [
        bench for bench in benches if isinstance(bench, IREEKernelBenchmark)
    ]
    compilation_results = {}
    if iree_benches:
        compile_results = batch_compile_iree_benches(
            iree_benches,
            callback=compile_callback,
            verbose=verbose,
            unique_id=unique_ids,
        )
        for bench, (config, vmfb_path, success) in zip(iree_benches, compile_results):
            compilation_results[id(bench)] = (vmfb_path, success)

    # Validate numerics
    if validate_numerics:
        compiled_benches = [
            bench
            for bench in benches
            if not isinstance(bench, IREEKernelBenchmark)
            or compilation_results[id(bench)]
        ]
        validation_results = {
            id(compiled_benches[i]): val_res
            for i, val_res in enumerate(batch_validate(benches, device, verbose))
        }
    else:
        validation_results = {id(bench): True for bench in benches}

    # Run benchmarks
    results = [None] * len(benches)
    all_bench_items = [(i, bench) for i, bench in enumerate(benches)]
    for i, bench in tqdm(
        all_bench_items, desc="Benchmarking kernels", disable=bench_callback is not None
    ):
        try:
            if isinstance(bench, IREEKernelBenchmark):
                vmfb_path, compile_success = compilation_results[id(bench)]
                numerical_success = validation_results[id(bench)]
                if compile_success and vmfb_path and numerical_success:
                    result = bench.bench_vmfb(
                        vmfb_path, device, num_iterations, timeout
                    )
                else:
                    result = bench.get_bench_result(0, False)
            else:
                if validation_results[id(bench)]:
                    result = bench.run_bench(device, num_iterations, timeout)
                else:
                    result = bench.get_bench_result(0, False)

            results[i] = result

        except Exception as e:
            if verbose:
                get_logger().error(
                    f"Benchmarking failed for {bench.config.get_name()} on backend {bench.backend}: {e}"
                )
            result = bench.get_bench_result(0, False)
            results[i] = result

        finally:
            if bench_callback:
                bench_callback(results[i])

    # Return results
    return results
