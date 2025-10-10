from dataclasses import asdict
import os

from kernel_bench.core.template import IREEKernelBenchmark
from kernel_bench.utils.iree_utils import *
from kernel_bench.config.types.attention import AttentionConfigBMNK
from ..attention_utils import (
    IREEAttentionTuningSpec,
    IntrinsicType,
    get_iree_attention_shapes,
)
from typing import Optional, override


class IREEVanillaAttentionBenchmark(IREEKernelBenchmark):
    config: AttentionConfigBMNK

    def _generate_attention_mlir_iree(
        self,
        config: AttentionConfigBMNK,
        tuning: Optional[IREEAttentionTuningSpec] = None,
    ):
        input_dtype = self.device_ctx.dtype_to_iree(config.dtype)
        query_shape, key_shape, value_shape, output_shape = get_iree_attention_shapes(
            config, self.device_ctx
        )

        shapes = f"""\
!dtype = {input_dtype}
!Q     = tensor<{query_shape}>
!K     = tensor<{key_shape}>
!V     = tensor<{value_shape}>
!O     = tensor<{output_shape}>
"""

        spec = ""
        if tuning and input_dtype == "f16":
            spec = f"""\
#tuning = {tuning.get_compilation_info()}
"""

        attn_kernel = f"""
#Q = affine_map<(b, m, n, k1, k2) -> (b, m, k1)>
#K = affine_map<(b, m, n, k1, k2) -> (b, k2, k1)>
#V = affine_map<(b, m, n, k1, k2) -> (b, k2, n)>
#S = affine_map<(b, m, n, k1, k2) -> ()>
#O = affine_map<(b, m, n, k1, k2) -> (b, m, n)>

func.func @main(%Q : !Q, %K : !K, %V : !V) -> !O {{
%scale = arith.constant 1.0 : !dtype
%empty = tensor.empty() : !O
%O = iree_linalg_ext.attention
    {{ indexing_maps = [#Q, #K, #V, #S, #O]
        ,decomposition_config = {{
        qk_attrs = {{attention_qk_matmul, lowering_config = {tuning.get_qk_config_info()}}},
        pv_attrs = {{attention_pv_matmul, lowering_config = {tuning.get_pv_config_info()}}}
        }}
        {",compilation_info = #tuning" if tuning and input_dtype == "f16" else ""}
    }}
    ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype) outs(%empty : !O) {{
        ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
        }} -> !O
return %O : !O
}}
"""
        mlir_template = shapes + "\n" + spec + "\n" + attn_kernel
        return mlir_template

    @override
    def compile_to_vmfb(self, mlir_path, vmfb_path):
        config = self.config
        spec = IREEAttentionTuningSpec(
            [1, 128, 0, 0, 0],
            [0, 0, 0, 0, 32],
            4,
            1,
            IntrinsicType.VMFMA_F32_32x32x16_F16,
            2,
            True,
        )

        mlir_content = self._generate_attention_mlir_iree(config, spec)

        # Write MLIR content to file
        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        # TODO: Do not hardcode device information, instead pass it as a class
        # Compile MLIR to vmfb
        exec_args = [
            "iree-compile",
            # Input file
            f"{mlir_path}",
            # Output file
            "-o",
            f"{vmfb_path}",
            # Target Device: hip
            "--iree-hal-target-device=hip",
            # Device: MI300x
            f"--iree-hip-target={self.device_ctx.hip_target}",
        ]
        if self.path_config.dumps:
            dump_file = self.path_config.dump_for(
                "iree", config.get_name() + ".debug.mlir"
            )
            phase_dump = self.path_config.dumps / "iree" / config.get_name()
            exec_args.append(f"--dump-compilation-phases-to={phase_dump}")

        ret_value, stdout, stderr = run_iree_command(exec_args)

        if ret_value == 0:
            if stderr and self.path_config.dumps:
                with open(dump_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
        else:
            if self.path_config.dumps:
                error_file = self.path_config.dump_for(
                    "iree", "log", config.get_name() + "_error.txt"
                )
                self.logger.error(
                    f"Failed to compile {mlir_path}. Error dumped in {error_file}"
                )
                with open(error_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
            else:
                self.logger.error(
                    f"Failed to compile {mlir_path}. No dump directory specified."
                )

            return False

        return True

    @override
    def get_runtime_args(self):
        query_shape, key_shape, value_shape, _ = get_iree_attention_shapes(
            self.config, self.device_ctx
        )
        runtime_args = [
            f"--input={shape}" for shape in [query_shape, key_shape, value_shape]
        ]
        runtime_args += ["--function=main"]
        return runtime_args
