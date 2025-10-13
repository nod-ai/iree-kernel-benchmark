from kernel_bench.config.types.attention import AttentionAttributes
from kernel_bench.config.types.attention.extend_attention_config import (
    attention_attributes_to_extend,
)


def get_extend_attention_configs():

    shapes = [
        AttentionAttributes(
            batch_size=1,
            num_seqs=1,
            context_len=seq_len,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_query_heads=128,
            num_kv_heads=128,
            head_size=128,
            head_size_kv=128,
            block_size=64,
            dtype="f16",
        )
        for seq_len in [1024, 2048, 4096]
    ]

    return [("sahil", attention_attributes_to_extend(shape)) for shape in shapes]
