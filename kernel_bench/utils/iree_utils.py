import os
import logging
import subprocess
from typing import Optional, Sequence
from typing import List, Tuple
import iree.runtime as ireert
from kernel_bench.utils.bench_utils import unit_to_microseconds
from kernel_bench.utils.print_utils import get_logger


def bench_kernel_ireert(
    vmfb_filename: os.PathLike,
    iree_args: List[str],
    num_iterations: int = 3,
    device: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[float, bool]:

    # print(
    #     f'iree-benchmark-module --device={device} --module={vmfb_filename} {" ".join(iree_args)}'
    # )

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
        get_logger().error(e)
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
        return 0, False

    benchmark_mean_time_us = sum(times) / float(len(times))
    return benchmark_mean_time_us, True


def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    (
        stdout_v,
        stderr_v,
    ) = (
        proc.stdout,
        proc.stderr,
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout, proc.stderr
    logging.getLogger().error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stdout, proc.stderr
