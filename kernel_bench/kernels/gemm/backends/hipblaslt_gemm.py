from typing import override, Tuple, Optional
import subprocess
import re

from kernel_bench.core.template import KernelBenchmark
from ..gemm_utils import GemmConfig


class HipBLASLtGemmBenchmark(KernelBenchmark):
    config: GemmConfig

    def _parse_output(self, output: str) -> Optional[float]:
        # Look for the average time line
        avg_time_pattern = r"Average time:\s*([\d.]+)\s*μs"
        match = re.search(avg_time_pattern, output)

        if match:
            return float(match.group(1))

        avg_time_pattern_alt = r"Average time:\s*([\d.]+)\s*us"
        match = re.search(avg_time_pattern_alt, output)

        if match:
            return float(match.group(1))

        return None

    @override
    def run_bench(self, device, num_iterations=1):
        debug = False
        config = self.config

        cmd = [
            "./hipblaslt/build/gemm_benchmark",
            "-M",
            str(config.M),
            "-N",
            str(config.N),
            "-K",
            str(config.K),
        ]

        if config.tA == "T":
            cmd.append("-tA")
        if config.tB == "T":
            cmd.append("-tB")

        input_dtype = config.operand_element_type
        output_dtype = config.result_element_type

        cmd.extend(
            [
                "-input_dtype",
                input_dtype,
                "-output_dtype",
                output_dtype,
            ]
        )

        try:
            # Run the executable
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                if debug:
                    self.logger.error(
                        f"Executable failed with return code {result.returncode}"
                    )
                    self.logger.error(f"stderr: {result.stderr}")
                    self.logger.error(f"stdout: {result.stdout}")
                return 0.0, False

            # Parse the output
            mean_time_us = self._parse_output(result.stdout)

            if mean_time_us is None:
                if debug:
                    self.logger.error("Failed to parse average time from output")
                    self.logger.error(f"stdout: {result.stdout}")
                return 0.0, False

            return mean_time_us, True

        except subprocess.TimeoutExpired:
            if debug:
                self.logger.error("Benchmark timed out")
            return 0.0, False
        except Exception as e:
            if debug:
                self.logger.error(f"Error running benchmark: {e}")
            return 0.0, False
