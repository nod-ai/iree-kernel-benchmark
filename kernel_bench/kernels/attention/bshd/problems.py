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

    shapes_paper = [
        ("Llama-3.1-8B", (1, 32, 642, 642, 128)),
        ("Llama-2-13b-hf", (1, 40, 706, 706, 128)),
        ("Llama-3.3-70B-Instruct", (1, 64, 642, 642, 128)),
        ("Mistral-7B-Instruct", (1, 32, 706, 706, 128)),
        ("sdxl-base-1", (2, 20, 1024, 1024, 64)),
        ("sdxl-base-2", (2, 20, 1024, 77, 64)),
    ]

    configs = []
    configs += [(tag, paper_to_bshd(shape, "f16")) for tag, shape in shapes_paper]
    return configs


def get_bshd_attention_configs() -> list[tuple[str, AttentionConfigBSHD]]:
    configs = []
    configs += cai_attn()
    configs += paper_attn()
    return configs
