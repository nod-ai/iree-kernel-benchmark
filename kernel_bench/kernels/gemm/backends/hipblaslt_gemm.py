from typing import override, Tuple, Optional
import subprocess
import re
import csv
from io import StringIO

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.paths import PathConfig, clear_dir
from ..gemm_utils import GemmConfig


class HipBLASLtGemmBenchmark(KernelBenchmark):
    config: GemmConfig

    @override
    def run_bench(self, device, num_iterations, timeout):
        if device.startswith("hip://"):
            device = int(device.split("hip://")[1])
        else:
            device = 0

        cmd = get_hipblaslt_cmd(self.config, device)
        cmds = [
            cmd
            # thread_trace_hipblaslt_cmd(cmd, self.config, self.path_config),
        ]

        bench_result = None
        for cmd in cmds:
            # self.logger.info(" ".join(cmd))
            bench_result = self._run_cmd(cmd)

        return bench_result

    def _run_cmd(self, cmd: list[str]):
        try:
            # Run the executable
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                self.logger.error(
                    (
                        f"Executable failed with return code {result.returncode}"
                        f"- stderr: \n{result.stderr}"
                        f"- stdout: \n{result.stdout}"
                    )
                )
                return self.get_bench_result(0.0, False)

            # self.logger.info(result.stdout)

            # Parse the output
            mean_time_us = parse_hipblaslt_us(result.stdout)
            hyperparams = parse_hipblaslt_block_sizes(result.stdout)

            if mean_time_us is None:
                self.logger.error(
                    (
                        "Failed to parse average time from output"
                        f"- stdout: \n{result.stdout}"
                    )
                )
                return self.get_bench_result(0.0, False)

            result = self.get_bench_result(mean_time_us, True)
            result.tuning_config = hyperparams
            return result

        except subprocess.TimeoutExpired:
            self.logger.error("Benchmark timed out")
            return self.get_bench_result(0.0, False)
        except Exception as e:
            self.logger.error(f"Error running benchmark: {e}")
            return self.get_bench_result(0.0, False)


def parse_hipblaslt_us(output: str, row_index: Optional[int] = None) -> float:
    lines = output.splitlines()

    # Precompile regex for header lines of the form "[0]:..." or "[123]:..."
    header_re = re.compile(r"^\s*\[(\d+)\]:\s*(.+)$")

    for i, line in enumerate(lines):
        m = header_re.match(line)
        if not m:
            continue

        idx_str, header_csv = m.groups()
        idx = int(idx_str)

        # If a specific row_index is requested, skip headers that don't match
        if row_index is not None and idx != row_index:
            continue

        # Parse header columns robustly with csv
        header_reader = csv.reader(StringIO(header_csv.strip()))
        headers = [h.strip() for h in next(header_reader, [])]
        if not headers:
            continue

        # Find index of 'us' column
        try:
            us_idx = headers.index("us")
        except ValueError:
            # This header doesn't contain 'us'; keep searching
            continue

        # Find the first data line after the header
        j = i + 1
        while j < len(lines):
            data_line = lines[j].strip()
            j += 1

            # Skip empty lines and metadata lines
            if not data_line or data_line.startswith("--") or data_line.startswith("["):
                continue

            # Heuristic: treat lines with commas as CSV data rows
            if "," in data_line:
                data_reader = csv.reader(StringIO(data_line))
                row = [v.strip() for v in next(data_reader, [])]
                if len(row) <= us_idx:
                    raise ValueError(
                        "Data row shorter than header; cannot retrieve 'us' field."
                    )

                # Convert and return the 'us' field
                try:
                    return float(row[us_idx])
                except ValueError as e:
                    raise ValueError(
                        f"Failed to parse 'us' value as float: {row[us_idx]}"
                    ) from e

        # If we reached here, we didn't find a data row for this header; keep searching

    raise ValueError(
        "Could not find a header with 'us' column and a following data row in the output."
    )


def parse_hipblaslt_block_sizes(
    output: str, solution_index: Optional[int] = None
) -> dict[str, int]:
    lines = output.splitlines()

    # If a specific header index is requested, narrow the search to the block after that header.
    if solution_index is not None:
        header_prefix = f"[{solution_index}]:"
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith(header_prefix):
                start_idx = i
                break
        if start_idx is None:
            raise ValueError(f"Header block [{solution_index}]: not found in output.")
        search_range = lines[start_idx + 1 :]
    else:
        search_range = lines

    # Find the solution name line
    sol_name = None
    for line in search_range:
        s = line.strip()
        if s.startswith("--Solution name:"):
            parts = s.split(":", 1)
            if len(parts) == 2:
                sol_name = parts[1].strip()
                break

    if not sol_name:
        raise ValueError("Solution name not found in output.")

    # Robustly match the block size token:
    # MT<M>x<N>x<K> followed by either an underscore or the end of the string.
    m = re.search(r"MT(\d+)x(\d+)x(\d+)(?:_|$)", sol_name)
    if not m:
        raise ValueError("Block size token 'MT<M>x<N>x<K>' not found in solution name.")

    block_m, block_n, block_k = map(int, m.groups())
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
    }


def thread_trace_hipblaslt_cmd(
    cmd: list[str], config: GemmConfig, path_config: PathConfig
):
    dump_dir = (
        path_config.dump_dir_for("hipblaslt") / "thread_trace" / config.get_name()
    )
    clear_dir(dump_dir)
    rocprofv3_prefix = [
        "rocprofv3",
        "--att",
        "--att-library-path",
        "/root/rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux/opt/rocm/lib/",
        "--att-target-cu",
        "0",
        "--kernel-include-regex",
        "Cijk",
        "--kernel-iteration-range",
        "200",
        "-d",
        f"{dump_dir}",
        "--output-format",
        "csv",
        "--",
    ]
    profile_cmd = rocprofv3_prefix + cmd
    return profile_cmd


def perf_counter_hipblaslt_cmd(
    cmd: list[str], config: GemmConfig, path_config: PathConfig
):
    dump_dir = (
        path_config.dump_dir_for("hipblaslt") / "perf_counter" / config.get_name()
    )
    rocprofv3_prefix = [
        "rocprofv3",
        "-T",
        "-d",
        f"{dump_dir}",
        "-o",
        "run",
        "-i",
        "cnt.txt",
        "--",
    ]
    thread_trace_cmd = rocprofv3_prefix + cmd
    return thread_trace_cmd


def get_hipblaslt_cmd(
    config: GemmConfig, device_id=None, tune=False, solution_index=None
):
    tA = "N" if config.tA == "T" else "T"
    tB = "N" if config.tB == "T" else "T"

    cmd = [
        "hipblaslt-bench",
        "--function",
        "matmul",
        "--transA",
        tA,
        "--transB",
        tB,
        "--a_type",
        f"{config.dtype}_r",
        "--b_type",
        f"{config.dtype}_r",
        "--c_type",
        f"{config.dtype}_r",
        "--d_type",
        f"{config.dtype}_r",
        "--scale_type",
        "f32_r",
        "--bias_type",
        "f32_r",
        "--compute_type",
        "f32_r",
        "--sizem",
        str(config.M),
        "--sizen",
        str(config.N),
        "--sizek",
        str(config.K),
        "--lda",
        str(config.M),
        "--ldb",
        str(config.K),
        "--ldc",
        str(config.M),
        "--ldd",
        str(config.M),
        "--initialization",
        "zero",
        "--alpha",
        "1",
        "--beta",
        "0",
        "--iters",
        "200",
        "--cold_iters",
        "200",
        "--use_gpu_timer",
        "--print_kernel_info",
        "--rotating",
        "0",
        "--device",
        str(device_id or 0),
    ]

    if tune or solution_index:
        cmd += [
            "--algo_method",
            "index",
            "--api_method",
            "cpp",
        ]

    if solution_index:
        cmd += [
            "--solution_index",
            str(solution_index),
        ]

    return cmd
