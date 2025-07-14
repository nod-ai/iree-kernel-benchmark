from dataclasses import dataclass
from typing import Union, Optional
import math
from abc import ABC, abstractmethod

@dataclass
class AttentionConfigBase(ABC):
    """Base class for attention configurations with common interface"""
    dtype: str
    
    @abstractmethod
    def get_name(self) -> str:
        """Get a descriptive name for this configuration"""
        pass
    
    @abstractmethod
    def get_query_shape(self) -> str:
        """Get the shape string for query tensor"""
        pass
    
    @abstractmethod
    def get_key_shape(self) -> str:
        """Get the shape string for key tensor"""
        pass
    
    @abstractmethod
    def get_value_shape(self) -> str:
        """Get the shape string for value tensor"""
        pass
    
    @abstractmethod
    def get_output_shape(self) -> str:
        """Get the shape string for output tensor"""
        pass
    
    @abstractmethod
    def get_byte_count(self) -> int:
        """Get total byte count for all tensors"""
        pass
    
    @abstractmethod
    def get_flops(self) -> int:
        """Get FLOP count for attention computation"""
        pass
    
    def _get_bytes_per_element(self) -> int:
        """Helper method to get bytes per element for the dtype"""
        dtype_bits_map = {
            "f32": 32,
            "f16": 16,
            "bf16": 16,
            "f8E4M3FNUZ": 8,
            "i8": 8,
            "i32": 32,
        }
        return dtype_bits_map[self.dtype] // 8


@dataclass
class AttentionConfigBMNK(AttentionConfigBase):
    """BMNK1K2 format: B=batch, M=query_seq, N=kv_embed_dim, K1=query_embed_dim, K2=kv_seq"""
    B: int
    M: int
    N: int
    K1: int
    K2: int
    dtype: str

    def get_name(self) -> str:
        return f"attention_bmnk1k2_{self.B}x{self.M}x{self.N}x{self.K1}x{self.K2}x{self.dtype}"

    def get_query_shape(self) -> str:
        return f"{self.B}x{self.M}x{self.K1}x{self.dtype}"

    def get_key_shape(self) -> str:
        return f"{self.B}x{self.K2}x{self.K1}x{self.dtype}"

    def get_value_shape(self) -> str:
        return f"{self.B}x{self.K2}x{self.N}x{self.dtype}"

    def get_output_shape(self) -> str:
        return f"{self.B}x{self.M}x{self.N}x{self.dtype}"

    def get_byte_count(self) -> int:
        dtype_bits_map = {
            "f32": 32,
            "f16": 16,
            "bf16": 16,
            "f8E4M3FNUZ": 8,
            "i8": 8,
            "i32": 32,
        }
        bytes_per_element = dtype_bits_map[self.dtype] // 8
        element_count = (
            (self.B * self.M * self.K1)
            + (self.B * self.K2 * self.K1)
            + (self.B * self.K2 * self.N)
            + (self.B * self.M * self.N)
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        # We measure flops of the two matmuls only
        qk_matmul_flops = 2 * self.B * self.M * self.K2 * self.K1
        pv_matmul_flops = 2 * self.B * self.M * self.N * self.K2
        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops


@dataclass
class AttentionConfigBSHD:
    """BSHD format: B=batch, S=seq_len, H=num_heads, D=head_dim"""
    B: int
    S: int
    H: int
    D: int
    dtype: str
    H_KV: int = None
    
    def __post_init__(self):
        if self.H_KV is None:
            self.H_KV = self.H

    def get_name(self) -> str:
        if self.H_KV == self.H:
            return f"attention_bshd_{self.B}x{self.S}x{self.H}x{self.D}x{self.dtype}"
        else:
            return f"attention_bshd_{self.B}x{self.S}x{self.H}x{self.D}_kv{self.H_KV}x{self.dtype}"

    def get_query_shape(self) -> str:
        return f"{self.B}x{self.S}x{self.H}x{self.D}x{self.dtype}"

    def get_key_shape(self) -> str:
        return f"{self.B}x{self.S}x{self.H_KV}x{self.D}x{self.dtype}"

    def get_value_shape(self) -> str:
        return f"{self.B}x{self.S}x{self.H_KV}x{self.D}x{self.dtype}"

    def get_output_shape(self) -> str:
        return f"{self.B}x{self.S}x{self.H}x{self.D}x{self.dtype}"

    def get_byte_count(self) -> int:
        dtype_bits_map = {
            "f32": 32,
            "f16": 16,
            "bf16": 16,
            "f8E4M3FNUZ": 8,
            "i8": 8,
            "i32": 32,
        }
        bytes_per_element = dtype_bits_map[self.dtype] // 8
        element_count = (
            (self.B * self.S * self.H * self.D)
            + (self.B * self.S * self.H_KV * self.D)
            + (self.B * self.S * self.H_KV * self.D)
            + (self.B * self.S * self.H * self.D)
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        qk_matmul_flops = 2 * self.B * self.S * self.S * self.H * self.D
        pv_matmul_flops = 2 * self.B * self.S * self.S * self.H * self.D
        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops

@dataclass
class AttentionAttributes:
    """Unified attributes for all attention types"""
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    head_size_kv: int
    batch_size: Optional[int] = None
    dtype: str = "f16"
    # -----------------------
    # Prefill specific
    num_seqs: Optional[int] = None
    max_seq_len: Optional[int] = None
    total_seq_len: Optional[int] = None
    context_len: Optional[int] = None
    fixed_seq_len_prefix: Optional[int] = None
    fixed_seq_len_extend: Optional[int] = None
    # -----------------------
    # Vanilla attention
    query_seq_len: Optional[int] = None
    kv_seq_len: Optional[int] = None
    # -----------------------
    # Decode specific
    block_size: Optional[int] = None

    def to_bmnk1k2(self) -> AttentionConfigBMNK:
        """Convert to BMNK1K2 format
        
        Attempts to infer sequence lengths from available attributes.
        Priority order: query_seq_len/kv_seq_len > context_len > max_seq_len > total_seq_len
        """
        if self.batch_size is None:
            raise ValueError("batch_size is required for BMNK1K2 conversion")
        
        return AttentionConfigBMNK(
            B=self.num_query_heads,
            M=self.query_seq_len,
            N=self.head_size_kv,
            K1=self.head_size,
            K2=self.kv_seq_len,
            dtype=self.dtype
        )

    def to_bshd(self) -> AttentionConfigBSHD:
        """Convert to BSHD format"""
        if self.batch_size is None:
            raise ValueError("batch_size is required for BSHD conversion")

        seq_len = self._get_query_seq_len()
        
        if self.head_size != self.head_size_kv:
            raise ValueError(
                f"BSHD format requires head_size == head_size_kv, "
                f"but got {self.head_size} != {self.head_size_kv}"
            )
        
        return AttentionConfigBSHD(
            B=self.batch_size,
            S=seq_len,
            H=self.num_query_heads,
            D=self.head_size,
            H_KV=self.num_kv_heads,
            dtype=self.dtype
        )

    def _get_query_seq_len(self) -> int:
        """Get query sequence length from available attributes"""
        if self.query_seq_len is not None:
            return self.query_seq_len
        elif self.context_len is not None:
            return self.context_len
        elif self.max_seq_len is not None:
            return self.max_seq_len
        elif self.total_seq_len is not None:
            return self.total_seq_len
        elif self.fixed_seq_len_prefix is not None:
            return self.fixed_seq_len_prefix
        elif self.fixed_seq_len_extend is not None:
            return self.fixed_seq_len_extend
        else:
            raise ValueError(
                "Cannot determine query sequence length. Please provide one of: "
                "query_seq_len, context_len, max_seq_len, total_seq_len, "
                "fixed_seq_len_prefix, or fixed_seq_len_extend"
            )

    def _get_kv_seq_len(self) -> int:
        """Get KV sequence length from available attributes"""
        if self.kv_seq_len is not None:
            return self.kv_seq_len
        elif self.query_seq_len is not None:
            return self.query_seq_len
        elif self.context_len is not None:
            return self.context_len
        elif self.max_seq_len is not None:
            return self.max_seq_len
        elif self.total_seq_len is not None:
            return self.total_seq_len
        elif self.fixed_seq_len_prefix is not None:
            return self.fixed_seq_len_prefix
        elif self.fixed_seq_len_extend is not None:
            return self.fixed_seq_len_extend
        else:
            raise ValueError(
                "Cannot determine KV sequence length. Please provide one of: "
                "kv_seq_len, query_seq_len, context_len, max_seq_len, total_seq_len, "
                "fixed_seq_len_prefix, or fixed_seq_len_extend"
            )

    def get_attention_type(self) -> str:
        """Infer the attention type based on available attributes"""
        if self.block_size is not None:
            return "decode"
        elif any([self.num_seqs, self.max_seq_len, self.total_seq_len, 
                 self.context_len, self.fixed_seq_len_prefix, self.fixed_seq_len_extend]):
            return "prefill"
        elif self.query_seq_len is not None or self.kv_seq_len is not None:
            return "vanilla"
        else:
            return "unknown"

def bmnk1k2_to_attention_attributes(config_bmnk: AttentionConfigBMNK) -> AttentionAttributes:
    """Convert BMNK1K2 format back to AttentionAttributes"""
    
    return AttentionAttributes(
        num_query_heads=config_bmnk.B,
        num_kv_heads=config_bmnk.B,
        head_size=config_bmnk.K1,
        head_size_kv=config_bmnk.N,
        batch_size=1,
        query_seq_len=config_bmnk.M,
        kv_seq_len=config_bmnk.K2,
        dtype=config_bmnk.dtype
    )


def bshd_to_attention_attributes(config_bshd: AttentionConfigBSHD) -> AttentionAttributes:
    """Convert BSHD format back to AttentionAttributes"""
    
    return AttentionAttributes(
        num_query_heads=config_bshd.H,
        num_kv_heads=config_bshd.H_KV,
        head_size=config_bshd.D,
        head_size_kv=config_bshd.D,
        batch_size=config_bshd.B,
        query_seq_len=config_bshd.S,
        kv_seq_len=config_bshd.S,
        dtype=config_bshd.dtype
    )

if __name__ == "__main__":
    print("=== Testing AttentionAttributes ===")
    
    vanilla_attrs = AttentionAttributes(
        num_query_heads=16,
        num_kv_heads=4,
        head_size=64,
        head_size_kv=64,
        batch_size=2,
        query_seq_len=1024,
        kv_seq_len=1024,
        dtype="f16"
    )
    
    print(f"Vanilla attention type: {vanilla_attrs.get_attention_type()}")
    
    vanilla_bmnk = vanilla_attrs.to_bmnk1k2()
    vanilla_bshd = vanilla_attrs.to_bshd()
    
    print(f"BMNK1K2: {vanilla_bmnk.get_name()}")
    print(f"BSHD: {vanilla_bshd.get_name()}")
    
    prefill_attrs = AttentionAttributes(
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        head_size_kv=128,
        batch_size=1,
        context_len=2048,
        max_seq_len=4096,
        dtype="bf16"
    )
    
    print(f"\nPrefill attention type: {prefill_attrs.get_attention_type()}")
    
    prefill_bmnk = prefill_attrs.to_bmnk1k2()
    prefill_bshd = prefill_attrs.to_bshd()
    
    print(f"BMNK1K2: {prefill_bmnk.get_name()}")
    print(f"BSHD: {prefill_bshd.get_name()}")
    
    decode_attrs = AttentionAttributes(
        num_query_heads=16,
        num_kv_heads=16,
        head_size=64,
        head_size_kv=64,
        batch_size=4,
        block_size=256,
        query_seq_len=1,
        kv_seq_len=1024,
        dtype="f32"
    )
    
    print(f"\nDecode attention type: {decode_attrs.get_attention_type()}")
    
    decode_bmnk = decode_attrs.to_bmnk1k2()
    
    print(f"BMNK1K2: {decode_bmnk.get_name()}")
    print(f"Query shape: {decode_bmnk.get_query_shape()}")
    print(f"Key shape: {decode_bmnk.get_key_shape()}")
    print(f"Value shape: {decode_bmnk.get_value_shape()}")