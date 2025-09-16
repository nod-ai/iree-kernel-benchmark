from abc import ABC, abstractmethod
import math
import os
import traceback
from sympy import Symbol
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.wave import LaunchableWave

from .utils import (
    BenchmarkResult,
    get_kernel_perf_stats,
    redirect_stderr_to_file,
    OpConfig,
)
from kernel_bench.utils.device_utils import machine_to_hip_target
from kernel_bench.utils.iree_utils import bench_kernel_ireert
from kernel_bench.tuning.hyperparam.parameters import (
    TuningParameter,
    TuningSpec,
    ParameterSymbol,
)
import sympy as sp


@dataclass
class KernelBenchmark(ABC):
    tag: str
    backend: str
    kernel_type: str
    machine: str
    config: OpConfig

    def __post_init__(self):
        self._tuning_spec = TuningSpec()
        self._param_symbols = {}
        self.setup_parameters()

    def add_param(self, name: str, bounds, **kwargs) -> ParameterSymbol:
        """Register and return a parameter symbol"""
        param = TuningParameter(name, bounds, **kwargs)
        self._tuning_spec.add_parameter(name, param)

        # Create a ParameterSymbol that wraps SymPy Symbol
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

    def get_bench_result(self, runtime_us: float, ok: bool):
        arithmetic_intensity, tflops_per_second = get_kernel_perf_stats(
            self.config, runtime_us if ok else math.inf
        )

        tuning_config = self._tuning_spec.to_dict() or None

        return BenchmarkResult(
            machine=self.machine,
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

    @abstractmethod
    def run_bench(
        self, device: str, num_iterations: int = 1, timeout: Optional[float] = None
    ) -> BenchmarkResult:
        pass


@dataclass
class IREEKernelBenchmark(KernelBenchmark):
    kernel_dir: Path
    dump_dir: Optional[Path]

    def __post_init__(self):
        super().__post_init__()
        self.machine = self.machine.upper()
        self.target = machine_to_hip_target(self.machine)

    @abstractmethod
    def compile_to_vmfb(self, mlir_path: Path, vmfb_path: Path) -> bool:
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
            self.config.get_runtime_args(self.backend),
            num_iterations,
            device,
            timeout,
        )
        return self.get_bench_result(runtime_us, ok)

    def run_bench(self, device, num_iterations=1, timeout=None):
        local_kernel_dir = self.kernel_dir / self.kernel_type / self.backend
        mlir_dir = local_kernel_dir / "mlir"
        vmfb_dir = local_kernel_dir / "vmfb"

        os.makedirs(mlir_dir, exist_ok=True)
        os.makedirs(vmfb_dir, exist_ok=True)

        mlir_path = mlir_dir / f"{self.config.get_name()}.mlir"
        vmfb_path = vmfb_dir / f"{self.config.get_name()}.vmfb"

        compile_success = self.compile_to_vmfb(mlir_path, vmfb_path)
        if not compile_success:
            return self.get_bench_result(0, False)

        return self.bench_vmfb(vmfb_path, device, num_iterations, timeout)


@dataclass
class WaveKernel:
    launchable: LaunchableWave
    hyperparams: Dict[Symbol, Any]
    dynamic_symbols: List[Symbol] = field(default_factory=list)


@dataclass
class WaveKernelBenchmark(IREEKernelBenchmark):
    @abstractmethod
    def load_wave_kernel(self) -> WaveKernel:
        pass

    @abstractmethod
    def get_compile_options(self) -> WaveCompileOptions:
        pass

    def compile_to_vmfb(self, mlir_path, vmfb_path):
        try:
            kernel = self.load_wave_kernel()
            compile_options = self.get_compile_options()

            compile_options.create_vmfb_file = vmfb_path
            compile_options.subs = kernel.hyperparams
            compile_options.dynamic_symbols = kernel.dynamic_symbols
            compile_options.iree_launch_async = False
            compile_options.run_bench = False
            compile_options.backend = "rocm"
            compile_options.target = self.target

            if self.dump_dir:
                dump_file = (
                    self.dump_dir / "wave" / (self.config.get_name() + ".debug.mlir")
                )
                with redirect_stderr_to_file(dump_file):
                    compile_options.mlir_print_ir_after_all = True
                    result = wave_compile(compile_options, kernel.launchable)
            else:
                result = wave_compile(compile_options, kernel.launchable)

            with open(mlir_path, "w") as mlir_out:
                mlir_out.write(result.asm)

            return True

        except Exception as e:
            print(f"Failed to compile {self.config.get_name()}: {e}")
            traceback.print_exception(e)
            return False
