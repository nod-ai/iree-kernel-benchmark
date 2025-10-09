from kernel_bench.config.types.attention import AttentionAttributes
from kernel_bench.config.types.attention.extend_attention_config import (
    attention_attributes_to_extend,
)


def get_extend_attention_configs():

    shapes = [
        AttentionAttributes(
            batch_size=1,
            num_seqs=2,
            context_len=1024,
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            head_size_kv=128,
            block_size=64,
        ),
        AttentionAttributes(
            batch_size=1,
            num_seqs=1,
            context_len=1024,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=64,
            head_size_kv=64,
            block_size=128,
        ),
        AttentionAttributes(
            batch_size=1,
            num_seqs=4,
            context_len=1024,
            num_query_heads=4,
            num_kv_heads=4,
            head_size=256,
            head_size_kv=256,
            block_size=64,
        ),
        AttentionAttributes(
            batch_size=1,
            num_seqs=1,
            context_len=1024,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=512,
            head_size_kv=512,
            block_size=32,
            dtype="f16",
        ),
    ]

    return [("sahil", attention_attributes_to_extend(shape)) for shape in shapes]
