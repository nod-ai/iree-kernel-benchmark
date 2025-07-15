from .attention_config import *


def get_attention_attrs_bmnk(
    B: int, M: int, N: int, K1: int, K2: int, dtype: str
) -> list[AttentionAttributes]:
    return bmnk1k2_to_attention_attributes(
        config_bmnk=AttentionConfigBMNK(dtype, B, M, N, K1, K2)
    )


def llm_sweep(dtype: str) -> list[AttentionAttributes]:
    configs = []
    # Batch sweep (batch * num_heads)
    for B in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192]:
        # M, K2 sweep (seq length)
        for M in [1024, 2048, 4096, 8192, 16384]:
            K2 = M
            # K1, N sweep (head dim)
            for N in [64, 128]:
                K1 = N
                configs.append(get_attention_attrs_bmnk(B, M, N, K1, K2, dtype))
    return configs


def sdxl_unet_sweep(dtype: str) -> list[AttentionAttributes]:
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
        configs.append(get_attention_attrs_bmnk(B, M, N, K1, K2, dtype))
    return configs


def bert_attn_sweep(dtype: str) -> list[AttentionAttributes]:
    configs = []
    sdxl_attn_shapes = [
        (12, 384, 64, 64, 384),
        (768, 4096, 64, 64, 64),
    ]
    for B, M, N, K1, K2 in sdxl_attn_shapes:
        configs.append(get_attention_attrs_bmnk(B, M, N, K1, K2, dtype))
    return configs


def llama3_405b_attn_sweep(dtype: str) -> list[AttentionAttributes]:
    configs = []
    for M in [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]:
        K2 = M
        configs.append(get_attention_attrs_bmnk(512, M, 128, 128, K2, dtype))
        M += 128
    return configs


def cai_attn(dtype: str) -> list[AttentionAttributes]:
    configs = []
    for M in [12288, 16384, 4145, 8192, 8698, 425, 8641, 8589, 4504]:
        configs.append(
            AttentionAttributes(
                num_seqs=1,
                num_query_heads=32,
                num_kv_heads=1,
                head_size=256,
                head_size_kv=256,
                batch_size=1,
                query_seq_len=M,
                kv_seq_len=M,
                dtype=dtype,
            )
        )
    return configs


type ConfigList = list[tuple[str, AttentionAttributes]]


def get_attention_configs() -> ConfigList:
    configs: ConfigList = []
    llm_configs = llm_sweep("f16")
    # llm_configs += llm_sweep("f8E4M3FNUZ")
    sdxl_configs = sdxl_unet_sweep("f16")
    # sdxl_configs += sdxl_unet_sweep("f8E4M3FNUZ")
    bert_configs = bert_attn_sweep("f16")
    # bert_configs += bert_attn_sweep("f8E4M3FNUZ")
    llama3_configs = llama3_405b_attn_sweep("f16")
    # llama3_configs += llama3_405b_attn_sweep("f8E4M3FNUZ")

    configs += [("llm", x) for x in llm_configs]
    configs += [("sdxl_unet", x) for x in sdxl_configs]
    configs += [("bert", x) for x in bert_configs]
    configs += [("llama3_405b", x) for x in llama3_configs]

    return configs


def get_attention_configs_gqa() -> ConfigList:
    cai_configs = cai_attn("bf16")

    configs = [("cai", x) for x in cai_configs]

    return configs
