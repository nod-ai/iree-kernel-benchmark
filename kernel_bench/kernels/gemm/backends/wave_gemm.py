from dataclasses import replace
from math import ceil
import traceback
from typing import override
import torch
from torch.testing import assert_close
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.iree_utils import generate_iree_ref

from kernel_bench.tuning.hyperparam import (
    CategoricalBounds,
    IntegerBounds,
)
from kernel_bench.core.template import WaveKernelBenchmark, WaveTemplate
from kernel_bench.utils.device_utils import (
    dtype_to_bits,
    dtype_to_bytes,
    dtype_to_torch,
)
from ..gemm_utils import GemmConfig


class WaveGemmBenchmark(WaveKernelBenchmark):
    config: GemmConfig

    def validate_config(self):
        input_dtype = self.config.operand_element_type
        if input_dtype == "f32" or "f8" in input_dtype:
            return False

        variant = self.config.tA + self.config.tB
        if variant != "NT":
            return False

        return True

    def setup_parameters(self):
        dtype = dtype_to_torch(self.config.operand_element_type)
        bitwidth = dtype_to_bits(self.config.operand_element_type)

        if bitwidth == 8:
            mfma_options = [(MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K8_F16)]
        elif dtype == torch.bfloat16 and self.target == "gfx950":
            mfma_options = [
                MMAType.F32_32x32x16_BF16,
                MMAType.F32_16x16x32_BF16,
            ]
        else:
            mfma_options = [
                MMAType.F32_16x16x16_F16,
                MMAType.F32_32x32x8_F16,
                MMAType.F32_32x32x16_K8_F16,
            ]
            if self.target == "gfx950":
                mfma_options = [
                    MMAType.F32_32x32x16_F16,
                    MMAType.F32_16x16x32_F16,
                ] + mfma_options

        self.mfma_variant = self.add_param(
            "MFMA_VARIANT",
            CategoricalBounds(mfma_options),
            initial_value=0,
            include_hyperparam=False,
        )
        self.BLOCK_M = self.add_param(
            "BLOCK_M",
            IntegerBounds(min=16, max=self.config.M, step=4),
            initial_value=128,
        )
        self.BLOCK_N = self.add_param(
            "BLOCK_N",
            IntegerBounds(min=16, max=self.config.N, step=2),
            initial_value=256,
        )
        self.BLOCK_K = self.add_param(
            "BLOCK_K",
            IntegerBounds(min=16, max=self.config.K, step=1),
            initial_value=64,
        )

        max_wg_m = ceil(self.config.M / 16) - 1
        self.GROUP_SIZE_M = self.add_param(
            "GROUP_SIZE_M",
            IntegerBounds(min=1, max=max_wg_m, step=1),
            initial_value=min(8, max_wg_m),
        )

        bytes_per_el = dtype_to_bytes(self.config.operand_element_type)
        shared_memory_constraint = (
            (self.BLOCK_M + 4) * self.BLOCK_K + (self.BLOCK_N + 4) * self.BLOCK_K
        ) * bytes_per_el - 65536
        self.add_constraint(shared_memory_constraint, "shared_memory_limit")

        num_wg_m = sympy.ceiling(self.config.M / self.BLOCK_M)
        group_size_constraint = self.GROUP_SIZE_M - num_wg_m
        self.add_constraint(group_size_constraint, "group_size_limit")

    @override
    def load_wave_kernel(self):
        config = self.config

        input_dtype = dtype_to_torch(config.operand_element_type)
        output_dtype = dtype_to_torch(config.result_element_type)
        quant_dtype = None
        if "f8" in config.operand_element_type:
            quant_dtype = input_dtype
            input_dtype = dtype_to_torch("f16")

        tA, tB = config.tA, config.tB

        base_gemm, hyperparams = get_reordered_matmul(
            config.M,
            config.N,
            config.K,
            self.BLOCK_M.value,
            self.BLOCK_N.value,
            self.BLOCK_K.value,
            self.GROUP_SIZE_M.value,
            mfma_variant=self.mfma_variant.value,
            # input_dtype=input_dtype,
            # output_dtype=output_dtype,
            # quantized_dtype=quant_dtype,
            # tA=tA,
            # tB=tB,
        )

        hyperparams.update(get_default_scheduling_params())

        return WaveTemplate(launchable=base_gemm, hyperparams=hyperparams)

    @override
    def extra_compile_options(self):
        use_scheduling = SchedulingType.PREFETCH

        return WaveCompileOptions(
            canonicalize=True,
            schedule=use_scheduling,
            use_buffer_ops=True,
            multi_buffer_count=(
                2
                if use_scheduling in [SchedulingType.FOUR_STAGE, SchedulingType.MODULO]
                else None
            ),
        )

    @override
    def validate_numerics(self, device):
        try:
            a = device_randn(self.config.M, self.config.K, dtype=torch.float16)
            b = device_randn(self.config.N, self.config.K, dtype=torch.float16)
            c = device_zeros(self.config.M, self.config.N, dtype=torch.float32)
            iree_ref = device_zeros(self.config.M, self.config.N, dtype=torch.float32)
        except Exception as e:
            self.logger.warn(
                f"Failed to allocate input tensors on device {device}",
                "".join(traceback.format_exception(e)),
            )
            return True

        try:
            kernel = self.load_wave_kernel()
            options = self.get_compile_options(kernel)
            gemm = wave_compile(options, kernel.launchable)
            gemm(a, b, c)

            generate_iree_ref("mmt", [a, b], [iree_ref], options)
            assert_close(c, iree_ref, check_device=False)
            return True

        except AssertionError as e:
            self.logger.error(
                f"Numerical accuracy failed for {self.config.get_name()} on backend {self.backend}",
                f"{e}",
            )
            return False
        except Exception as e:
            self.logger.warn(
                f"Could not validate numerics for {self.config.get_name()} on backend {self.backend}",
                "".join(traceback.format_exception(e)),
            )
            return True
