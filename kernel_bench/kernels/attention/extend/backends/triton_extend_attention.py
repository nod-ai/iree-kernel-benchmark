from aiter.ops.triton.extend_attention import extend_attention_fwd

from kernel_bench.core.template import KernelBenchmark
from kernel_bench.config.types.attention import AttentionConfigExtend
from kernel_bench.utils.torch_utils import benchmark_function_torch


class TritonExtendAttentionBenchmark(KernelBenchmark):
    config: AttentionConfigExtend

    def run_bench(self, device, num_iterations, timeout=None):
        inputs = self.config.get_inputs()

        try:
            mean_time_us = benchmark_function_torch(
                extend_attention_fwd,
                iterations=num_iterations,
                compile=True,
                # Extend attention inputs
                q_extend=inputs.q_extend,
                k_extend=inputs.k_extend,
                v_extend=inputs.v_extend,
                o_extend=inputs.output,
                k_buffer=inputs.k_buffer,
                v_buffer=inputs.v_buffer,
                qo_indptr=inputs.qo_indptr,
                kv_indptr=inputs.kv_indptr,
                kv_indices=inputs.kv_indices,
                custom_mask=None,
                is_causal=None,
                mask_indptr=None,
                max_len_extend=inputs.max_len_extend,
                logit_cap=inputs.logit_cap,
                config={
                    "BLOCK_M": 16,
                    "BLOCK_N": 16,
                    "waves_per_eu": 0,
                    "num_warps": 4,
                    "matrix_instr_nonkdim": 16,
                    "kpack": 2,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Failed to benchmark kernel {self.config.get_name()}: {e}"
            )
            return self.get_bench_result(0, False)

        return self.get_bench_result(mean_time_us, True)
