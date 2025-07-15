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
    """BSHD format: B=num_seqs, H=num_query_heads, H_KV=num_kv_heads, N_Q=query_seq_len, D_KV=head_size_kv, D_Q=head_size, N_KV=kv_seq_len"""
    B: int      # num_seqs
    H: int      # num_query_heads
    H_KV: int   # num_kv_heads
    N_Q: int    # query_seq_len
    D_KV: int   # head_size_kv
    D_Q: int    # head_size
    N_KV: int   # kv_seq_len
    dtype: str

    def get_name(self) -> str:
        return f"attention_bshd_{self.B}x{self.H}x{self.H_KV}x{self.N_Q}x{self.D_KV}x{self.D_Q}x{self.N_KV}x{self.dtype}"

    def get_query_shape(self) -> str:
        return f"{self.B}x{self.N_Q}x{self.H}x{self.D_Q}x{self.dtype}"

    def get_key_shape(self) -> str:
        return f"{self.B}x{self.N_KV}x{self.H_KV}x{self.D_Q}x{self.dtype}"

    def get_value_shape(self) -> str:
        return f"{self.B}x{self.N_KV}x{self.H_KV}x{self.D_KV}x{self.dtype}"

    def get_output_shape(self) -> str:
        return f"{self.B}x{self.N_Q}x{self.H}x{self.D_KV}x{self.dtype}"

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
            (self.B * self.N_Q * self.H * self.D_Q)      # Query
            + (self.B * self.N_KV * self.H_KV * self.D_Q)  # Key
            + (self.B * self.N_KV * self.H_KV * self.D_KV) # Value
            + (self.B * self.N_Q * self.H * self.D_KV)      # Output
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        # QK matmul: (B, N_Q, H, D_Q) x (B, N_KV, H_KV, D_Q) -> (B, H, N_Q, N_KV)
        # Assuming H_KV is broadcast to H for computation
        qk_matmul_flops = 2 * self.B * self.H * self.N_Q * self.N_KV * self.D_Q
        
        # PV matmul: (B, H, N_Q, N_KV) x (B, N_KV, H_KV, D_KV) -> (B, N_Q, H, D_KV)
        # Assuming H_KV is broadcast to H for computation
        pv_matmul_flops = 2 * self.B * self.H * self.N_Q * self.N_KV * self.D_KV
        
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
        if self.num_seqs is None:
            raise ValueError("num_seqs is required for BSHD conversion")
        
        query_seq_len = self._get_query_seq_len()
        kv_seq_len = self._get_kv_seq_len()
        
        return AttentionConfigBSHD(
            B=self.num_seqs,
            H=self.num_query_heads,
            H_KV=self.num_kv_heads,
            N_Q=query_seq_len,
            D_KV=self.head_size_kv,
            D_Q=self.head_size,
            N_KV=kv_seq_len,
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
        head_size=config_bshd.D_Q,
        head_size_kv=config_bshd.D_KV,
        num_seqs=config_bshd.B,
        query_seq_len=config_bshd.N_Q,
        kv_seq_len=config_bshd.N_KV,
        dtype=config_bshd.dtype
    )
