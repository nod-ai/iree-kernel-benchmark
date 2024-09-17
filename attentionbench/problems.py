import itertools

SDXL_ATTN = [
    (2, 10, 4096, 4096, 64, "f16"),
    (2, 10, 4096, 64, 64, "f16"),
    (2, 10, 1024, 1024, 64, "f16"),
    (2, 20, 1024, 64, 64, "f16"),
]

def generate_attention_shapes(
    configs,
    name : str,
    batch_sizes : list[int], 
    head_counts : list[int], 
    head_dims : list[int], 
    seq_lengths : list[int], 
    datatypes : list[str]):

    for B, H, S_Q, S_KV, DH, datatype in itertools.product(batch_sizes, head_counts, seq_lengths, seq_lengths, head_dims, datatypes):
        bytes = B * H * 2 * (2 * S_KV * DH + 2 * S_Q * DH + S_Q * S_KV)
        if bytes < 1e9:
            configs.append((name, B, H, S_Q, S_KV, DH, datatype))

def llama70battention(configs):
    generate_attention_shapes(
        configs,
        "llama70battention",
        batch_sizes=[1, 2, 4],
        head_counts=[32, 40, 64],
        head_dims=[128],
        seq_lengths=[1024, 2048, 4096],
        datatypes=["f16"],
    )

def sdxlattention(configs):
    for B, H, S_Q, S_KV, DH, _ in SDXL_ATTN:
        bytes = B * H * 2 * (2 * S_KV * DH + 2 * S_Q * DH + S_Q * S_KV)
        if bytes < 1e9:
            for datatype in ["f16"]:
                configs.append(("sdxlattention", B, H, S_Q, S_KV, DH, datatype))

def flash_attention(configs):
    # batch_sizes = [1, 2, 4, 8, 16, 32]
    # head_counts = [12, 24, 36, 42, 48]
    # head_dims = [32, 64, 128]
    # seq_lengths = [64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64320]
    # datatypes = ["f16"]

    # shapes = []
    # shapes = [
    #     (1, 42, 384, 64320, 64, "f16"),
    #     (1, 42, 4096, 4096, 64, "f16"),
    #     (1, 42, 384, 4096, 64, "f16"),
    #     (1, 42, 8192, 8192, 64, "f16"),
    #     (1, 42, 384, 8192, 64, "f16"),
    #     (1, 42, 16384, 16384, 64, "f16"),
    #     (1, 42, 384, 16384, 64, "f16"),
    # ]
    
    # yield from generate_attention_shapes("generalattention", batch_sizes, head_counts, head_dims, seq_lengths, datatypes)
    llama70battention(configs)
    sdxlattention(configs)
