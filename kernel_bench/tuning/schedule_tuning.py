import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.fx as fx

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.optimize_schedule import (
    OptimizationAlgorithm,
    OptimizationResult,
    ScheduleOptimizer,
)
from wave_lang.kernel.wave.scheduling.resources import (
    Operation,
    get_custom_operation_type,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.scheduling.verifier import (
    ScheduleValidator as ScheduleModifier,
)
from wave_lang.kernel.wave.tuner.utils import (
    enum_to_str,
    format_latency_us,
    latency_to_us,
)
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.print_utils import dump_schedule, load_schedule

from wave_lang.kernel.wave.utils.run_utils import get_default_arch

from kernel_bench.config.base import OpConfig


@dataclass
class TimingResult:
    """Results from timing a kernel execution."""

    latency_ms: float
    throughput_tflops: float
    trace_file: Optional[str] = None


def calculate_throughput(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    latency_seconds: float,
) -> float:
    """Calculate theoretical throughput in TFLOPs.

    Args:
        batch_size: Batch dimension
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        head_dim: Head dimension
        latency_seconds: Execution time in seconds

    Returns:
        Throughput in TFLOPs
    """
    # For attention, we have:
    # 1. QK matmul: 2 * B * H * M * N * K operations
    # 2. Softmax: 2 * B * H * M * N operations
    # 3. V matmul: 2 * B * H * M * N * K operations
    # Total: 4 * B * H * M * N * K + 2 * B * H * M * N operations
    total_ops = (
        4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
        + 2 * batch_size * num_heads * seq_len_q * seq_len_k
    )
    return total_ops / (latency_seconds * 1e12)  # Convert to TFLOPs


def measure_kernel_runtime(
    config: OpConfig,
    compile_kernel_func: Callable,
    bench_kernel_func: Callable,
    kernel_dir: Path,
    schedule_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
    iteration: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Measure the latency of vanilla attention kernel with given schedule.

    Args:
        config: Attention configuration
        schedule_file: Optional path to schedule file to use. If None, uses default schedule
        log_dir: Optional directory to save RPD traces
        iteration: Optional iteration number for trace file naming

    Returns:
        Average latency in microseconds
    """

    config_name = config.get_name()

    mlir_dir = kernel_dir / "mlir"
    vmfb_dir = kernel_dir / "vmfb"
    os.makedirs(mlir_dir, exist_ok=True)
    os.makedirs(vmfb_dir, exist_ok=True)

    mlir_path = mlir_dir / f"{config_name}.mlir"
    vmfb_path = vmfb_dir / f"{config_name}.vmfb"

    try:
        compile_success = compile_kernel_func(
            config,
            mlir_path,
            vmfb_path,
            extra_compiler_args=(
                {"override_schedule": schedule_file} if schedule_file else None
            ),
        )
        if not compile_success:
            raise Exception("Compilation Failure")
    except:
        logger.warning(f"Failed to compile {config_name}")
        return float("inf")

    try:
        bench_success = runtime_us, bench_success = bench_kernel_func(
            config, vmfb_path, num_iterations=1
        )
        if not bench_success:
            raise Exception("Runtime Failure")
    except:
        logger.warning(f"Failed to benchmark {config_name}")
        return float("inf")

    return runtime_us


def setup_logging(config: OpConfig, run_name: str) -> Tuple[logging.Logger, Path]:
    """Set up logging for the tuning process.

    Args:
        config: Attention configuration

    Returns:
        Tuple of (logger, log_dir)
    """

    # Create directories
    base_dir = Path("attention_tuning")
    log_dir = base_dir / f"tune_{run_name}" / config.get_name()
    schedules_dir = log_dir / "schedules"

    # Create directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    schedules_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = logging.getLogger("attention_tuner")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Disable propagation to root logger to prevent duplication
    logger.propagate = False

    # File handler for detailed log
    log_file = log_dir / "tuning.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial configuration
    def enum_to_str(obj):
        if isinstance(obj, Enum):
            return str(obj)
        if isinstance(obj, tuple):
            return tuple(enum_to_str(x) for x in obj)
        return obj

    config_dict = {k: enum_to_str(v) for k, v in asdict(config).items()}
    logger.info("Starting attention kernel tuning")
    logger.info(f"Configuration:\n{json.dumps(config_dict, indent=2)}")

    return logger, log_dir


def save_schedule(
    schedule: Dict,
    latency: float,
    iteration: int,
    schedules_dir: Path,
    logger: logging.Logger,
    graph: Optional[fx.Graph] = None,
    initiation_interval: Optional[int] = None,
    num_stages: Optional[int] = None,
    resource_reservations: Optional[np.ndarray] = None,
    resource_names: Optional[list[str]] = None,
    original_schedule_file: Optional[str] = None,
) -> None:
    """Save a schedule to a JSON file and optionally to a schedule file.

    Args:
        schedule: The schedule to save
        latency: The latency achieved with this schedule
        iteration: The iteration number
        schedules_dir: Directory to save schedules
        logger: Logger instance
        graph: Optional FX graph for schedule file format
        initiation_interval: Optional initiation interval for schedule file format
        num_stages: Optional number of stages for schedule file format
        resource_reservations: Optional resource reservations for schedule file format
        resource_names: Optional resource names for schedule file format
        original_schedule_file: Optional original schedule file to preserve structure
    """
    # Save in JSON format
    schedule_file = schedules_dir / f"schedule_{iteration:04d}.json"
    schedule_data = {
        "iteration": int(iteration),
        "latency_us": float(latency_to_us(latency)),
        "schedule": {str(k): int(v) for k, v in schedule.items()},
    }

    with open(schedule_file, "w") as f:
        json.dump(schedule_data, f, indent=2)

    # Save in schedule file format if all required parameters are provided
    if all([graph, initiation_interval, num_stages]):
        schedule_txt_file = schedules_dir / f"schedule_{iteration:04d}.txt"
        dump_schedule(
            schedule,
            initiation_interval,
            num_stages,
            str(schedule_txt_file),
            resource_reservations,
            resource_names,
            original_schedule_file,
        )
        logger.debug(f"Saved schedule for iteration {iteration} to {schedule_txt_file}")

    logger.debug(f"Saved schedule for iteration {iteration} to {schedule_file}")


class TuningLogger:
    """Custom logger for the optimization process."""

    def __init__(self, logger: logging.Logger, schedules_dir: Path):
        self.logger = logger
        self.schedules_dir = schedules_dir
        self.best_latency = float("inf")
        self.best_iteration = -1
        self.history = []
        self.current_iteration = 0
        # Store additional parameters for schedule saving
        self.graph = None
        self.initiation_interval = None
        self.num_stages = None
        self.resource_reservations = None
        self.resource_names = None
        self.original_schedule_file = None

    def set_schedule_params(
        self,
        graph: fx.Graph,
        initiation_interval: int,
        num_stages: int,
        resource_reservations: Optional[np.ndarray] = None,
        resource_names: Optional[list[str]] = None,
        original_schedule_file: Optional[str] = None,
    ) -> None:
        """Set parameters needed for saving schedules in schedule file format.

        Args:
            graph: FX graph for schedule file format
            initiation_interval: Initiation interval for schedule file format
            num_stages: Number of stages for schedule file format
            resource_reservations: Optional resource reservations for schedule file format
            resource_names: Optional resource names for schedule file format
            original_schedule_file: Optional original schedule file to preserve structure
        """
        self.graph = graph
        self.initiation_interval = initiation_interval
        self.num_stages = num_stages
        self.resource_reservations = resource_reservations
        self.resource_names = resource_names
        self.original_schedule_file = original_schedule_file

    def log_iteration(
        self, iteration: int, schedule: Dict, latency: float, is_improvement: bool
    ) -> None:
        """Log an optimization iteration and save the schedule.

        Args:
            iteration: Current iteration number
            schedule: Current schedule
            latency: Achieved latency
            is_improvement: Whether this is an improvement
        """
        # Always save the schedule, regardless of whether it's an improvement
        schedule_filename = f"schedule_{iteration:04d}"
        save_schedule(
            schedule,
            latency,
            iteration,
            self.schedules_dir,
            self.logger,
            self.graph,
            self.initiation_interval,
            self.num_stages,
            self.resource_reservations,
            self.resource_names,
            self.original_schedule_file,
        )

        self.history.append(
            {
                "iteration": int(iteration),
                "latency_us": float(latency_to_us(latency)),
                "is_improvement": bool(is_improvement),
                "schedule_file": str(f"{schedule_filename}.json"),
                "schedule_txt_file": str(f"{schedule_filename}.txt"),
            }
        )

        if is_improvement:
            self.best_latency = latency
            self.best_iteration = iteration
            self.logger.info(
                f"Iteration {iteration}: Found improvement! "
                f"Latency: {format_latency_us(latency)}"
            )
        else:
            self.logger.debug(
                f"Iteration {iteration}: No improvement. "
                f"Latency: {format_latency_us(latency)}"
            )

    def log_summary(self) -> None:
        """Log a summary of the tuning process."""
        self.logger.info("\nTuning Summary:")
        self.logger.info(f"Best latency: {format_latency_us(self.best_latency)}")
        self.logger.info(f"Best iteration: {self.best_iteration}")
        self.logger.info(f"Total iterations: {len(self.history)}")

        # Save history
        history_file = self.schedules_dir.parent / "tuning_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        self.logger.info(f"Tuning history saved to {history_file}")


def get_custom_operation_type_val(custom: "CustomOp") -> str:
    """Get the string value of the operation type for a custom operation.

    Args:
        custom: The custom operation to get the type value for

    Returns:
        The string value of the operation type (e.g. "read_shared", "write_global", etc.)
    """
    op_type = get_custom_operation_type(custom)
    return op_type.value if op_type is not None else Operation.NOOP.value


def create_optimizer_logger() -> logging.Logger:
    """Create a separate logger for the optimizer to avoid duplication."""
    optimizer_logger = logging.getLogger("attention_optimizer")
    optimizer_logger.setLevel(logging.INFO)
    optimizer_logger.handlers.clear()
    optimizer_logger.propagate = False

    # Add console handler for optimizer logging
    optimizer_console_handler = logging.StreamHandler()
    optimizer_console_handler.setLevel(logging.INFO)
    optimizer_formatter = logging.Formatter(
        "%(message)s"
    )  # Simpler format for optimizer
    optimizer_console_handler.setFormatter(optimizer_formatter)
    optimizer_logger.addHandler(optimizer_console_handler)

    return optimizer_logger


def create_validator(
    initial_schedule: Dict,
    initiation_interval: int,
    nodes: List,
    edges: List,
    resource_names: Optional[List[str]] = None,
) -> ScheduleModifier:
    """Create a schedule validator with common parameters."""
    return ScheduleModifier(
        initial_schedule=initial_schedule,
        T=initiation_interval,
        nodes=nodes,
        resource_limits=(
            np.array([2] * len(resource_names))
            if resource_names
            else np.array([2, 2, 2, 2, 2])
        ),
        node_rrt_getter=lambda node: (
            node.rrt
            if hasattr(node, "rrt")
            else np.zeros((1, len(resource_names) if resource_names else 5))
        ),
        raw_edges_list=edges,
        num_resource_types=len(resource_names) if resource_names else 5,
    )


def get_result_dict(
    config: OpConfig,
    initial_latency: float,
    initial_schedule: Dict,
    result: OptimizationResult,
    tuning_logger: TuningLogger,
) -> Tuple[Dict[str, Any], os.PathLike]:
    improvement_history = [
        float(latency_to_us(float(h))) for h in result.improvement_history
    ]

    # Convert config to JSON-serializable format
    config_dict = {k: enum_to_str(v) for k, v in asdict(config).items()}

    # Get the best iteration from tuning logger
    best_iteration = tuning_logger.best_iteration
    if best_iteration == 0:
        best_schedule_filename = "schedule_0000"
    else:
        best_schedule_filename = f"schedule_{best_iteration:04d}"

    final_results = {
        "config": config_dict,
        "initial_latency_us": float(latency_to_us(initial_latency)),
        "initial_schedule_file": "initial_schedule.txt",
        "best_latency_us": float(latency_to_us(result.latency)),
        "best_schedule_file": f"{best_schedule_filename}.txt",
        "best_schedule_json_file": f"{best_schedule_filename}.json",
        "best_iteration": int(best_iteration),
        "total_iterations": int(result.iterations),
        "improvement_history": improvement_history,
        "all_schedules": tuning_logger.history,
    }

    return final_results, best_schedule_filename


def save_final_results(
    config: OpConfig,
    initial_latency: float,
    initial_schedule: Dict,
    result: OptimizationResult,
    log_dir: Path,
    logger: logging.Logger,
    tuning_logger: TuningLogger,
) -> None:
    """Save final results to JSON file with references to schedule files."""
    final_results, best_schedule_filename = get_result_dict(
        config, initial_latency, initial_schedule, result, tuning_logger
    )

    results_file = log_dir / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Final results saved to {results_file}")
    logger.info(
        f"Best schedule saved to {log_dir / 'schedules' / f'{best_schedule_filename}.txt'}"
    )


def tune_kernel_schedule(
    config: OpConfig,
    run_name: str,
    load_kernel_func: Callable,
    compile_kernel_func: Callable,
    bench_kernel_func: Callable,
    kernel_dir: Path,
    max_iterations=100,
    extra_compile_options: dict[str, Any] = {},
    seed: int = None,
) -> Tuple[Dict, float]:
    """Tune the kernel using ScheduleOptimizer."""
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Set up logging
    logger, log_dir = setup_logging(config, run_name)
    tuning_logger = TuningLogger(logger, log_dir / "schedules")

    logger.warning("RPD tracer not available, falling back to CUDA events")

    base_kernel, hyperparams = load_kernel_func(config)

    # Update hyperparameters with scheduling parameters
    hyperparams.update(get_default_scheduling_params())

    # Compile the kernel once and dump the schedule
    config_name = config.get_name()

    mlir_dir = kernel_dir / "mlir"
    vmfb_dir = kernel_dir / "vmfb"
    os.makedirs(mlir_dir, exist_ok=True)
    os.makedirs(vmfb_dir, exist_ok=True)

    mlir_path = mlir_dir / f"{config_name}.mlir"
    vmfb_path = vmfb_dir / f"{config_name}.vmfb"

    initial_schedule_file = log_dir / "initial_schedule.txt"
    assert not initial_schedule_file.exists()
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.MODULO,
        canonicalize=True,
        create_vmfb_file=vmfb_path,
        run_bench=False,
        iree_launch_async=False,
        target=get_default_arch(),
        **extra_compile_options,
    )
    options.dump_schedule = initial_schedule_file
    compiled_kernel = wave_compile(options, base_kernel)
    with open("test.mlir", "w") as file:
        file.write(compiled_kernel.asm)
    assert initial_schedule_file.exists()

    # Create a graph to hold the schedule nodes
    graph = compiled_kernel.get_trace().region_graph.subgraphs["region_0"]

    # Load the schedule using print_utils
    (
        initial_schedule,
        initiation_interval,
        num_stages,
        nodes,
        edges,
        resource_reservations,
        resource_names,
    ) = load_schedule(initial_schedule_file, graph)

    logger.info(f"Loaded schedule with II={initiation_interval}, stages={num_stages}")
    logger.info(f"Found {len(nodes)} nodes and {len(edges)} edges")

    # Create validator with loaded schedule data
    validator = create_validator(
        initial_schedule, initiation_interval, nodes, edges, resource_names
    )

    # Get initial schedule and compiled kernel
    logger.info("Getting initial schedule...")
    initial_latency = measure_kernel_runtime(
        config,
        compile_kernel_func=compile_kernel_func,
        bench_kernel_func=bench_kernel_func,
        kernel_dir=kernel_dir,
        schedule_file=initial_schedule_file,
        log_dir=log_dir,
        iteration=0,
        logger=logger,
    )

    logger.info(f"Initial latency: {format_latency_us(initial_latency)}")

    # Set up tuning logger with schedule parameters
    tuning_logger.set_schedule_params(
        graph,
        initiation_interval,
        num_stages,
        resource_reservations,
        resource_names,
        initial_schedule_file,
    )

    # Save initial schedule
    save_schedule(
        initial_schedule,
        initial_latency,
        0,  # iteration 0 for initial schedule
        log_dir / "schedules",
        logger,
        graph,
        initiation_interval,
        num_stages,
        resource_reservations,
        resource_names,
        original_schedule_file=initial_schedule_file,
    )

    # Add initial schedule to tuning logger history
    tuning_logger.history.append(
        {
            "iteration": 0,
            "latency_us": float(latency_to_us(initial_latency)),
            "is_improvement": True,  # Initial schedule is considered an improvement
            "schedule_file": "schedule_0000.json",
            "schedule_txt_file": "schedule_0000.txt",
        }
    )
    tuning_logger.best_latency = initial_latency
    tuning_logger.best_iteration = 0

    # Create measurement function that captures config and compiled kernel
    def measure_fn(schedule: Dict) -> float:
        nonlocal initial_schedule_file
        # Save the schedule to a temporary file
        schedule_file = (
            log_dir
            / "schedules"
            / f"temp_schedule_{tuning_logger.current_iteration:04d}.txt"
        )
        dump_schedule(
            schedule,
            initiation_interval,
            num_stages,
            str(schedule_file),
            resource_reservations,
            resource_names,
            original_schedule_file=initial_schedule_file,
        )

        # Update initial_schedule_file to use the newly created schedule file
        initial_schedule_file = str(schedule_file)

        latency = measure_kernel_runtime(
            config,
            compile_kernel_func,
            bench_kernel_func,
            kernel_dir,
            schedule_file=str(schedule_file),
            log_dir=log_dir,
            iteration=tuning_logger.current_iteration,
            logger=logger,
        )
        return latency

    # Create a separate logger for the optimizer to avoid duplication with main logger
    optimizer_logger = create_optimizer_logger()

    # Create progress file path
    progress_file = log_dir / "tuning_progress.csv"

    # Create and run optimizer with its own logger
    optimizer = ScheduleOptimizer(
        validator=validator,
        measure_fn=measure_fn,
        algorithm=OptimizationAlgorithm.HILL_CLIMBING,
        logger=optimizer_logger,
        progress_file=str(progress_file),
        tuning_logger=tuning_logger,
    )

    # Run optimization
    result = optimizer.optimize(
        max_iterations=max_iterations, max_no_improvement=20, verbose=True
    )

    # Save final results
    save_final_results(
        config,
        initial_latency,
        initial_schedule,
        result,
        log_dir,
        logger,
        tuning_logger,
    )

    result_json, _ = get_result_dict(
        config, initial_latency, initial_schedule, result, tuning_logger
    )
    return result_json, result.latency
