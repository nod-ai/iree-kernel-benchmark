import os
import subprocess
from typing import Any, Optional, Sequence
from typing import List, Tuple
import iree.runtime as ireert
import torch
from kernel_bench.utils.bench_utils import unit_to_microseconds
from kernel_bench.utils.dtypes.device_context import DeviceContext
from kernel_bench.utils.print_utils import get_logger


def bench_kernel_ireert(
    vmfb_filename: os.PathLike,
    iree_args: List[str],
    num_iterations: int = 3,
    device: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[float, bool]:
    logger = get_logger()

    extra_flags = {}
    func_name = None
    inputs = []
    for flag in iree_args:
        split_key_value = flag[2:].split("=")
        key = split_key_value[0]
        value = "=".join(split_key_value[1:])
        if key == "function":
            func_name = value
            continue
        if key == "input":
            inputs.append(value)
            continue
        extra_flags[key] = value

    # command = f"iree-benchmark-module --module={vmfb_filename} --function={func_name} --device={device} "
    # for input in inputs:
    #     command += f"--input={input} "
    # logger.info(command)

    try:
        bench_results = ireert.benchmark.benchmark_module(
            vmfb_filename,
            entry_function=func_name,
            inputs=inputs,
            timeout=timeout,
            device=device,
            device_allocator="caching",
            benchmark_repetitions=num_iterations,
            **extra_flags,
        )
    except Exception as e:
        logger.error(f"Error benchmarking: {e}")
        return 0, False

    times = []
    for bench_result in bench_results:
        bench_name = bench_result.benchmark_name
        if bench_name.split("/")[-1] == "real_time":
            time_and_unit = bench_result.time.split(" ")
            assert (
                len(time_and_unit) == 2
            ), "expected the benchmark time to be the time and unit separated by a space."
            time_us = unit_to_microseconds(
                real_time=float(time_and_unit[0]),
                time_unit=time_and_unit[1],
            )
            times.append(time_us)

    if len(times) == 0:
        logger.error("Could not parse benchmark results")
        return 0, False

    benchmark_mean_time_us = sum(times) / float(len(times))
    return benchmark_mean_time_us, True


def run_iree_command(args: Sequence[str] = ()):
    logger = get_logger()
    # logger.info("Exec:", " ".join(args))
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout, proc.stderr
    logger.error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stdout, proc.stderr


def get_default_accumulator_element_type(operand_element_type: str) -> str:
    return {
        "f16": "f32",
        "bf16": "f32",
        "f32": "f32",
        "f8": "f32",
        "f8E4M3FNUZ": "f32",
        "f8E5M2FNUZ": "f32",
        "f8E4M3FN": "f32",
        "f8E5M2": "f32",
        "i8": "i32",
        "i32": "i32",
    }[operand_element_type]


def get_default_result_element_type(
    operand_element_type: str, raw_accumulators: bool
) -> str:
    return (
        get_default_accumulator_element_type(operand_element_type)
        if raw_accumulators
        else operand_element_type
    )


def shape_to_iree(
    shape: tuple[int, ...] | Any, dtype: str | torch.dtype, device_ctx: DeviceContext
) -> str:
    if isinstance(dtype, torch.dtype):
        dtype = device_ctx.dtype_from_torch(dtype).to_iree_string()
    else:
        dtype = device_ctx.dtype_to_iree(dtype)
    if isinstance(shape, tuple):
        return "x".join(map(str, [*shape, dtype]))
    else:
        return f"1x{dtype}"


def tensor_to_iree_shape(tensor: Any, device_ctx: DeviceContext) -> str:
    if isinstance(tensor, torch.Tensor):
        return shape_to_iree(tensor.shape, tensor.dtype, device_ctx)
    if isinstance(tensor, float):
        return "1xf32"
    if isinstance(tensor, int):
        return "1xi32"
    if isinstance(tensor, bool):
        return "1xbool"
