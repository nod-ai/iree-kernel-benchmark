from attention_utils import AttentionConfig


def llm_sweep(configs: list[AttentionConfig], dtype: str):
    # Batch sweep (batch * num_heads)
    for B in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192]:
        # M, K2 sweep (seq length)
        for M in [1024, 2048, 4096, 8192, 16384]:
            K2 = M
            # K1, N sweep (head dim)
            for N in [64, 128]:
                K1 = N
                configs.append(AttentionConfig(B, M, N, K1, K2, dtype))


def sdxl_unet_sweep(configs: list[AttentionConfig], dtype: str):
    sdxl_attn_shapes = [
        (20, 4096, 64, 64, 4096),
        (20, 4096, 64, 64, 64),
        (40, 1024, 64, 64, 1024),
        (40, 1024, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(AttentionConfig(B, M, N, K1, K2, dtype))


def bert_attn_sweep(configs: list[AttentionConfig], dtype: str):
    sdxl_attn_shapes = [
        (20, 4096, 64, 64, 4096),
        (20, 4096, 64, 64, 64),
        (40, 1024, 64, 64, 1024),
        (40, 1024, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(AttentionConfig(B, M, N, K1, K2, dtype))


def get_attention_configs() -> list:
    configs = []
    llm_sweep(configs, "f16")
    sdxl_unet_sweep(configs, "f16")
    bert_attn_sweep(configs, "f16")
    return configs
