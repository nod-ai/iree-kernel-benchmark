from kernel_bench.config.types.attention import AttentionConfigBMNK


def llm_sweep(dtype: str) -> list[AttentionConfigBMNK]:
    configs = []
    # Batch sweep (batch * num_heads)
    for B in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192]:
        # M, K2 sweep (seq length)
        for M in [1024, 2048, 4096, 8192, 16384]:
            K2 = M
            # K1, N sweep (head dim)
            for N in [64, 128]:
                K1 = N
                configs.append(AttentionConfigBMNK(B, M, N, K1, K2, dtype))
    return configs


def sdxl_unet_sweep(dtype: str) -> list[AttentionConfigBMNK]:
    configs = []
    sdxl_attn_shapes = [
        (1, 4096, 64, 64, 4096),
        (1, 4096, 64, 64, 64),
        (2, 1024, 64, 64, 1024),
        (2, 1024, 64, 64, 64),
        (4, 4096, 64, 64, 4096),
        (4, 4096, 64, 64, 64),
        (8, 1024, 64, 64, 1024),
        (8, 1024, 64, 64, 64),
        (20, 4096, 64, 64, 4096),
        (20, 4096, 64, 64, 64),
        (40, 1024, 64, 64, 1024),
        (40, 1024, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(AttentionConfigBMNK(B, M, N, K1, K2, dtype))
    return configs


def bert_attn_sweep(dtype: str) -> list[AttentionConfigBMNK]:
    configs = []
    sdxl_attn_shapes = [
        (12, 384, 64, 64, 384),
        (768, 4096, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(AttentionConfigBMNK(B, M, N, K1, K2, dtype))
    return configs


def llama3_405b_attn_sweep(dtype: str) -> list[AttentionConfigBMNK]:
    configs = []
    for M in [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]:
        K2 = M
        configs.append(AttentionConfigBMNK(512, M, 128, 128, K2, dtype))
        M += 128
    return configs


type ConfigList = list[tuple[str, AttentionConfigBMNK]]


def paper_attn() -> ConfigList:
    def to_bmnk(shape: tuple[int, ...], dtype: str):
        B, H, S_Q, S_K, D = shape
        return AttentionConfigBMNK(B=H, M=S_Q, N=D, K1=D, K2=S_K, dtype=dtype)

    shapes = [
        ("Llama-3.1-8B", (1, 32, 642, 642, 128)),
        ("Llama-2-13b-hf", (1, 40, 706, 706, 128)),
        ("Llama-3.3-70B-Instruct", (1, 64, 642, 642, 128)),
        ("Mistral-7B-Instruct", (1, 32, 706, 706, 128)),
        ("sdxl-base-1", (2, 20, 1024, 1024, 64)),
        ("sdxl-base-2", (2, 20, 1024, 77, 64)),
    ]

    configs = []
    configs += [(tag, to_bmnk(shape, "f16")) for tag, shape in shapes]
    return configs


def get_vanilla_attention_configs(use_fp8=True) -> ConfigList:
    configs: ConfigList = []
    llm_configs = llm_sweep("f16")
    sdxl_configs = sdxl_unet_sweep("f16")
    bert_configs = bert_attn_sweep("f16")
    llama3_configs = llama3_405b_attn_sweep("f16")

    if use_fp8:
        llm_configs += llm_sweep("f8")
        sdxl_configs += sdxl_unet_sweep("f8")
        bert_configs += bert_attn_sweep("f8")
        llama3_configs += llama3_405b_attn_sweep("f8")

    configs += [("llm", x) for x in llm_configs]
    configs += [("sdxl_unet", x) for x in sdxl_configs]
    configs += [("bert", x) for x in bert_configs]
    configs += [("llama3_405b", x) for x in llama3_configs]

    configs += paper_attn()

    return configs
