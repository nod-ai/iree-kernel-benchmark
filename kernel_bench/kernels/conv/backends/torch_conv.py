import torch
from typing import override

from kernel_bench.utils.device_utils import dtype_to_torch
from kernel_bench.core.template import KernelBenchmark
from kernel_bench.utils.torch_utils import benchmark_function_torch
from ..conv_utils import ConvConfig


class TorchConvBenchmark(KernelBenchmark):
    config: ConvConfig

    @override
    def run_bench(self, device, num_iterations, timeout):
        config = self.config

        operation = config.OP
        dtype = dtype_to_torch(config.input_dtype)

        batch_size = config.N
        filter_batch_size = config.F
        num_channels = config.C

        stride = config.S
        filter_height = config.P
        filter_width = config.Q

        output_height = config.H
        output_width = config.W
        input_height = output_height * stride + filter_height - 1
        input_width = output_width * stride + filter_width - 1

        if not "nchw" in operation:
            return 0, False

        input_shape = (batch_size, num_channels, input_height, input_width)
        weight_shape = (filter_batch_size, num_channels, filter_height, filter_width)

        input = torch.randn(input_shape, dtype=dtype, device="cuda")
        weight = torch.randn(weight_shape, dtype=dtype, device="cuda")

        try:
            mean_time_us = benchmark_function_torch(
                torch.nn.functional.conv2d,
                input,
                weight,
                bias=None,
                stride=stride,
                padding=0,
                dilation=1,
                groups=1,
                iterations=num_iterations,
            )

        except Exception as e:
            self.logger.error(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
