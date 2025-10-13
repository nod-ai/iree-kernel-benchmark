from kernel_bench.config.types.attention.bshd_attention_config import (
    AttentionConfigBSHD,
)


def cai_attn() -> list[AttentionConfigBSHD]:
    configs = []
    for dtype in ["f16", "bf16"]:
        for seq_len in [12288, 16384, 4145, 8192, 8698, 425, 8641, 8589, 4504]:
            configs.append(
                AttentionConfigBSHD(
                    B=1,
                    H=32,
                    H_KV=1,
                    D_Q=256,
                    D_KV=256,
                    N_Q=seq_len,
                    N_KV=seq_len,
                    dtype=dtype,
                )
            )
    return configs
