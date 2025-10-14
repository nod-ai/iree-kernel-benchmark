from kernel_bench.config.types.attention.bshd_attention_config import (
    AttentionConfigBSHD,
)


def cai_attn() -> list[tuple[str, AttentionConfigBSHD]]:
    shapes = []
    for dtype in ["f16", "bf16"]:
        for seq_len in [12288, 16384, 4145, 8192, 8698, 425, 8641, 8589, 4504]:
            shapes.append(
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
    return [("cai", shape) for shape in shapes]


def paper_attn() -> list[tuple[str, AttentionConfigBSHD]]:
    def paper_to_bshd(shape: tuple[int, ...], dtype: str):
        B, H, S_Q, S_K, D = shape
        return AttentionConfigBSHD(
            B=B, H=H, H_KV=H, N_Q=S_Q, D_KV=D, D_Q=D, N_KV=S_K, dtype=dtype
        )

    def bmnk_to_bshd(shape: tuple[int, ...], dtype: str):
        B, M, N, K1, K2 = shape
        return AttentionConfigBSHD(
            B=1, H=B, H_KV=B, N_Q=M, N_KV=K2, D_Q=K1, D_KV=N, dtype=dtype
        )

    shapes_paper = [
        ("Llama-3.1-8B", (1, 32, 642, 642, 128)),
        ("Llama-2-13b-hf", (1, 40, 706, 706, 128)),
        ("Llama-3.3-70B-Instruct", (1, 64, 642, 642, 128)),
        ("Mistral-7B-Instruct", (1, 32, 706, 706, 128)),
    ]

    shapes_sdxl = [
        ("sdxl-large", (8, 1024, 64, 64, 64)),
        ("sdxl-medium", (40, 1024, 64, 64, 1024)),
    ]

    configs = []
    configs += [(tag, paper_to_bshd(shape, "f16")) for tag, shape in shapes_paper]
    configs += [(tag, bmnk_to_bshd(shape, "f16")) for tag, shape in shapes_sdxl]
    return configs


def get_bshd_attention_configs() -> list[tuple[str, AttentionConfigBSHD]]:
    configs = []
    configs += cai_attn()
    configs += paper_attn()
    return configs
