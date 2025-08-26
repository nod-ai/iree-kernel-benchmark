from dataclasses import asdict
from ..utils import *
from pathlib import Path
from typing import Optional, override
from .conv_utils import ConvConfig
import traceback

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.templates.conv import get_igemm_conv2d
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_ones,
)

import torch


class TorchConvBenchmark(KernelBenchmark):
    def _clear_mem(self, *tensors):
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @override
    def bench_kernel(
        self, config: ConvConfig, vmfb_filename, num_iterations=3, debug=False
    ):
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

        input = device_randn(input_shape, dtype=dtype)
        weight = device_randn(weight_shape, dtype=dtype)

        self._clear_mem()
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iterations):
                torch.nn.functional.conv2d(
                    input,
                    weight,
                    bias=None,
                    stride=stride,
                    padding=0,
                    dilation=1,
                    groups=1,
                )
            end_event.record()
            torch.cuda.synchronize()

        except Exception as e:
            print(f"Failed to benchmark kernel {config.get_name()}: {e}")
            return 0, False
        self._clear_mem(input, weight)

        delta_time_ms = start_event.elapsed_time(end_event)
        delta_time_us = delta_time_ms * 1e3
        mean_time_us = delta_time_us / num_iterations

        return mean_time_us, True
