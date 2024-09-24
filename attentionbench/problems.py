from attention_utils import AttentionConfig


def llm_sweep(dtype: str) -> list[AttentionConfig]:
    configs = []
    # Batch sweep (batch * num_heads)
    for B in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192]:
        # M, K2 sweep (seq length)
        for M in [1024, 2048, 4096, 8192, 16384]:
            K2 = M
            # K1, N sweep (head dim)
            for N in [64, 128]:
                K1 = N
                configs.append(AttentionConfig(B, M, N, K1, K2, dtype))
    return configs


def sdxl_unet_sweep(dtype: str) -> list[AttentionConfig]:
    configs = []
    sdxl_attn_shapes = [
        (20, 4096, 64, 64, 4096),
        (20, 4096, 64, 64, 64),
        (40, 1024, 64, 64, 1024),
        (40, 1024, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(AttentionConfig(B, M, N, K1, K2, dtype))
    return configs


def bert_attn_sweep(dtype: str) -> list[AttentionConfig]:
    configs = []
    sdxl_attn_shapes = [
        (12, 384, 64, 64, 384),
        (768, 4096, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(AttentionConfig(B, M, N, K1, K2, dtype))
    return configs


def get_attention_configs() -> list[tuple[str, AttentionConfig]]:
    configs: list[tuple[str, AttentionConfig]] = []
    llm_configs = llm_sweep("f16")
    sdxl_configs = sdxl_unet_sweep("f16")
    bert_configs = bert_attn_sweep("f16")

    # configs += [("llm_sweep", x) for x in llm_configs]
    # configs += [("sdxl_unet_sweep", x) for x in sdxl_configs]
    configs += [("bert_attn_sweep", x) for x in bert_configs]

    return configs
