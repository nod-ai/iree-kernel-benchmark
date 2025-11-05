from dataclasses import replace
from math import ceil
import traceback
from typing import override
import torch
from torch.testing import assert_close
from kernel_bench.utils.iree_utils import shape_to_iree
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
from ..gemm_utils import GemmConfig


class WaveGemmBenchmark(WaveKernelBenchmark):
    config: GemmConfig

    def validate_config(self):
        config = self.config

        if config.M < 4 or config.N < 4 or config.K < 4:
            return False

        return True

    def setup_parameters(self):
        dtype_spec = self.device_ctx.resolve_dtype(self.config.dtype)
        bitwidth = dtype_spec.bitwidth()
        target = self.device_ctx.hip_target

        if bitwidth == 8:
            mfma_options = [(MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K8_F16)]
        elif dtype_spec.to_torch() == torch.bfloat16 and target == "gfx950":
            mfma_options = [
                MMAType.F32_16x16x32_BF16,
                MMAType.F32_32x32x16_BF16,
            ]
        else:
            mfma_options = [
                MMAType.F32_16x16x16_F16,
                MMAType.F32_32x32x8_F16,
                MMAType.F32_32x32x16_K8_F16,
            ]
            if target == "gfx950":
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
            IntegerBounds(min=4, max=min(256, self.config.M), step=4),
            initial_value=128,
            clamp_value=True,
        )
        self.BLOCK_N = self.add_param(
            "BLOCK_N",
            IntegerBounds(min=4, max=min(256, self.config.N), step=4),
            initial_value=256,
            clamp_value=True,
        )
        self.BLOCK_K = self.add_param(
            "BLOCK_K",
            IntegerBounds(min=4, max=min(256, self.config.K), step=4),
            initial_value=64,
            clamp_value=True,
        )

        max_wg_m = ceil(self.config.M / 16) - 1
        self.GROUP_SIZE_M = self.add_param(
            "GROUP_SIZE_M",
            IntegerBounds(min=1, max=max_wg_m, step=1),
            initial_value=16,
            clamp_value=True,
        )

        bytes_per_el = dtype_spec.num_bytes()
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
        if config.dtype == "f8":
            input_dtype = self.device_ctx.dtype_to_torch("f16")
            quantized_dtype = self.device_ctx.dtype_to_torch(config.dtype)
        else:
            input_dtype = self.device_ctx.dtype_to_torch(config.dtype)
            quantized_dtype = None

        base_gemm, hyperparams = get_reordered_matmul(
            self.config.M,
            self.config.N,
            self.config.K,
            self.BLOCK_M.value,
            self.BLOCK_N.value,
            self.BLOCK_K.value,
            self.GROUP_SIZE_M.value,
            mfma_variant=self.mfma_variant.value,
            input_dtype=input_dtype,
            quantized_dtype=quantized_dtype,
            tA=config.tA,
            tB=config.tB,
        )
        hyperparams.update(get_default_scheduling_params())
        return WaveTemplate(launchable=base_gemm, hyperparams=hyperparams)

    @override
    def extra_compile_options(self):
        use_scheduling = SchedulingType.PREFETCH
        supports_g2s = self.device_ctx.hip_target.startswith("gfx95")

        return WaveCompileOptions(
            canonicalize=True,
            schedule=use_scheduling,
            use_buffer_ops=True,
            use_global_to_shared=False,
            # multi_buffer_count=1,
            # postprocess=get_unroll_pipeline(1),
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

    @override
    def get_runtime_args(self):
        config = self.config

        shape_A = (config.K, config.M) if config.tA == "T" else (config.M, config.K)
        shape_B = (config.N, config.K) if config.tB == "T" else (config.K, config.N)
        shape_C = (config.M, config.N)

        input_dtype = "f16" if config.dtype == "f8" else config.dtype

        inp1 = shape_to_iree(shape_A, input_dtype, self.device_ctx)
        inp2 = shape_to_iree(shape_B, input_dtype, self.device_ctx)
        out = shape_to_iree(shape_C, "f32", self.device_ctx)

        runtime_args = [
            f"--input={inp1}",
            f"--input={inp2}",
            f"--input={out}",
            "--function=isolated_benchmark",
        ]
        return runtime_args


def get_unroll_pipeline(unroll_factor: int):
    return f"""
    module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
            %0 = transform.structured.match ops{{["scf.for"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.loop.unroll %0 {{ factor = {unroll_factor} }} : !transform.any_op
            transform.yield
        }}
    }}
    """
