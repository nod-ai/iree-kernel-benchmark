from dataclasses import dataclass
from typing import Any, Union, Optional, Literal
import torch
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

from kernel_bench.utils.bench_utils import OpConfig, change_shape_dtype
from kernel_bench.utils.device_utils import (
    dtype_to_bytes,
    dtype_to_torch,
    stringify_shape,
    stringify_tensor_shape,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
)


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


@dataclass
class AttentionConfigBMNK(OpConfig):
    B: int
    M: int
    N: int
    K1: int
    K2: int
    dtype: str
    attributes: AttentionAttributes = None

    def __post_init__(self):
        if not self.attributes:
            self.attributes = bmnk1k2_to_attention_attributes(self)

    def get_name(self) -> str:
        return f"attention_bmnk1k2_{self.B}x{self.M}x{self.N}x{self.K1}x{self.K2}x{self.dtype}"

    def get_query_shape(self) -> str:
        return stringify_shape((self.B, self.M, self.K1), self.dtype)

    def get_key_shape(self) -> str:
        return stringify_shape((self.B, self.K2, self.K1), self.dtype)

    def get_value_shape(self) -> str:
        return stringify_shape((self.B, self.K2, self.N), self.dtype)

    def get_output_shape(self) -> str:
        return stringify_shape((self.B, self.M, self.N), self.dtype)

    def get_byte_count(self) -> int:
        bytes_per_element = dtype_to_bytes(self.dtype)
        element_count = (
            (self.B * self.M * self.K1)
            + (self.B * self.K2 * self.K1)
            + (self.B * self.K2 * self.N)
            + (self.B * self.M * self.N)
        )
        byte_count = element_count * bytes_per_element
        return byte_count

    def get_flops(self) -> int:
        qk_matmul_flops = 2 * self.B * self.M * self.K2 * self.K1
        pv_matmul_flops = 2 * self.B * self.M * self.N * self.K2
        total_flops = qk_matmul_flops + pv_matmul_flops
        return total_flops

    def get_runtime_args(self, backend_name):
        query_shape = self.get_query_shape()
        key_shape = self.get_key_shape()
        value_shape = self.get_value_shape()

        if backend_name == "wave":
            inputs = [query_shape, key_shape, value_shape]
            if "f8" in self.dtype:
                inputs = [change_shape_dtype(shape, "f16") for shape in inputs]
            out_shape = change_shape_dtype(self.get_output_shape(), "f32")
            inputs.append(out_shape)
            bench_function = "isolated_benchmark"

        else:
            inputs = [query_shape, key_shape, value_shape]
            bench_function = "main"

        return [f"--input={input}" for input in inputs] + [
            f"--function={bench_function}"
        ]

    def to_dict(self):
        return {
            "B": self.B,
            "M": self.M,
            "N": self.N,
            "K1": self.K1,
            "K2": self.K2,
            "dtype": self.dtype,
        }


@dataclass
class AttentionConfigBSHD(OpConfig):
    B: int  # num_seqs
    H: int  # num_query_heads
    H_KV: int  # num_kv_heads
    N_Q: int  # query_seq_len
    D_KV: int  # head_size_kv
    D_Q: int  # head_size
    N_KV: int  # kv_seq_len
    dtype: str
    attributes: AttentionAttributes

    def __post_init__(self):
        if not self.attributes:
            self.attributes = bshd_to_attention_attributes(self)

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
        bytes_per_element = dtype_to_bytes(self.dtype)
        element_count = (
            (self.B * self.N_Q * self.H * self.D_Q)  # Query
            + (self.B * self.N_KV * self.H_KV * self.D_Q)  # Key
            + (self.B * self.N_KV * self.H_KV * self.D_KV)  # Value
            + (self.B * self.N_Q * self.H * self.D_KV)  # Output
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

    def get_runtime_args(self, backend_name):
        query_shape = self.get_query_shape()
        key_shape = self.get_key_shape()
        value_shape = self.get_value_shape()

        if backend_name.startswith("wave"):
            inputs = [query_shape, key_shape, value_shape]
            if "f8" in self.dtype:
                inputs = [change_shape_dtype(shape, "f16") for shape in inputs]
            out_shape = change_shape_dtype(self.get_output_shape(), "f32")
            inputs.append(out_shape)
            bench_function = "isolated_benchmark"

        else:
            inputs = [query_shape, key_shape, value_shape]
            bench_function = "main"

        return [f"--input={input}" for input in inputs] + [
            f"--function={bench_function}"
        ]

    def to_dict(self):
        return {
            "B": self.B,
            "H": self.H,
            "H_KV": self.H_KV,
            "N_Q": self.N_Q,
            "D_KV": self.D_KV,
            "D_Q": self.D_Q,
            "N_KV": self.N_KV,
            "dtype": self.dtype,
        }


class AttentionConfigExtend(AttentionConfigBSHD):
    inputs: "ExtendAttentionInputs" = None

    def __post_init__(self):
        super().__post_init__()

    def get_runtime_args(self, backend_name):
        if not self.inputs:
            self.get_inputs()
        bench_inputs = [
            stringify_shape(self.inputs.q_extend.shape, self.dtype),
            stringify_shape(self.inputs.k_extend.shape, self.dtype),
            stringify_shape(self.inputs.v_extend.shape, self.dtype),
            stringify_shape(self.inputs.k_buffer.shape, self.dtype),
            stringify_shape(self.inputs.v_buffer.shape, self.dtype),
            stringify_shape(self.inputs.qo_indptr.shape, "i32"),
            stringify_shape(self.inputs.kv_indptr.shape, "i32"),
            stringify_shape(self.inputs.kv_indices.shape, "i32"),
            stringify_shape(self.inputs.output.shape, "f32"),
            stringify_shape(self.inputs.max_len_extend, "i32"),
        ]
        bench_function = "isolated_benchmark" if backend_name == "wave" else "main"
        return [f"--input={input}" for input in bench_inputs] + [
            f"--function={bench_function}"
        ]

    def get_inputs(self):
        self.inputs = create_extend_attention_inputs(
            self.attributes, dtype_to_torch(self.dtype)
        )
        seq_len = self.inputs.max_len_extend
        self.attributes.max_seq_len = seq_len
        self.attributes.kv_seq_len = seq_len
        self.attributes.query_seq_len = seq_len
        self.N_Q = seq_len
        self.N_KV = seq_len
        return self.inputs


def bmnk1k2_to_attention_attributes(
    config_bmnk: AttentionConfigBMNK,
) -> AttentionAttributes:
    if config_bmnk.attributes:
        return config_bmnk.attributes
    return AttentionAttributes(
        num_query_heads=config_bmnk.B,
        num_kv_heads=config_bmnk.B,
        head_size=config_bmnk.K1,
        head_size_kv=config_bmnk.N,
        batch_size=1,
        query_seq_len=config_bmnk.M,
        kv_seq_len=config_bmnk.K2,
        dtype=config_bmnk.dtype,
    )


def bshd_to_attention_attributes(
    config_bshd: AttentionConfigBSHD,
) -> AttentionAttributes:
    if config_bshd.attributes:
        return config_bshd.attributes
    return AttentionAttributes(
        num_query_heads=config_bshd.H,
        num_kv_heads=config_bshd.H_KV,
        head_size=config_bshd.D_Q,
        head_size_kv=config_bshd.D_KV,
        num_seqs=config_bshd.B,
        max_seq_len=max(config_bshd.N_Q, config_bshd.N_KV),
        query_seq_len=config_bshd.N_Q,
        kv_seq_len=config_bshd.N_KV,
        dtype=config_bshd.dtype,
    )


def validate_obj_attrs(obj: Any, attrs: list[str]):
    try:
        for attr in attrs:
            if not obj.__getattribute__(attr):
                raise Exception()
    except:
        raise ValueError(f"Could not find attribute {attr} in {obj}")


def attention_attributes_to_bmnk1k2(
    shape: AttentionAttributes,
) -> AttentionConfigBMNK:
    validate_obj_attrs(
        shape,
        [
            "num_query_heads",
            "query_seq_len",
            "head_size_kv",
            "head_size",
            "kv_seq_len",
            "dtype",
        ],
    )
    return AttentionConfigBMNK(
        B=shape.num_query_heads,
        M=shape.query_seq_len,
        N=shape.head_size_kv,
        K1=shape.head_size,
        K2=shape.kv_seq_len,
        dtype=shape.dtype,
        attributes=shape,
    )


def attention_attributes_to_bshd(
    shape: AttentionAttributes,
) -> AttentionConfigBSHD:
    validate_obj_attrs(
        shape,
        [
            "num_seqs",
            "num_query_heads",
            "num_kv_heads",
            "query_seq_len",
            "head_size_kv",
            "head_size",
            "kv_seq_len",
            "dtype",
        ],
    )
    return AttentionConfigBSHD(
        B=shape.num_seqs,
        H=shape.num_query_heads,
        H_KV=shape.num_kv_heads,
        N_Q=shape.query_seq_len,
        D_KV=shape.head_size_kv,
        D_Q=shape.head_size,
        N_KV=shape.kv_seq_len,
        dtype=shape.dtype,
        attributes=shape,
    )


def attention_attributes_to_extend(
    shape: AttentionAttributes,
) -> AttentionConfigExtend:
    validate_obj_attrs(
        shape,
        [
            "num_seqs",
            "num_query_heads",
            "num_kv_heads",
            "head_size_kv",
            "head_size",
            "dtype",
        ],
    )
    return AttentionConfigExtend(
        B=shape.num_seqs,
        H=shape.num_query_heads,
        H_KV=shape.num_kv_heads,
        N_Q=shape.query_seq_len,
        D_KV=shape.head_size_kv,
        D_Q=shape.head_size,
        N_KV=shape.kv_seq_len,
        dtype=shape.dtype,
        attributes=shape,
    )


@dataclass
class ExtendAttentionInputs:
    q_extend: torch.Tensor
    k_extend: torch.Tensor
    v_extend: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    output: torch.Tensor
    max_len_extend: int
    logit_cap: float


def _generate_config_filename(
    shape: AttentionAttributes, dtype_str: str, seed: Optional[int] = None
) -> str:
    """
    Generate a unique filename based on attention configuration parameters.

    Args:
        shape: AttentionAttributes instance
        dtype_str: String representation of the tensor dtype
        seed: Random seed used for generation (if any)

    Returns:
        Unique filename string for this configuration
    """
    # Extract key parameters that define the configuration
    params = [
        f"nseqs{shape.num_seqs}",
        f"nqh{shape.num_query_heads}",
        f"nkvh{shape.num_kv_heads}",
        f"hs{shape.head_size}",
        f"ctx{shape.context_len}",
    ]

    # Add optional parameters if they exist
    if shape.fixed_seq_len_prefix is not None:
        params.append(f"prefixlen{shape.fixed_seq_len_prefix}")
    if shape.fixed_seq_len_extend is not None:
        params.append(f"extendlen{shape.fixed_seq_len_extend}")

    # Add dtype and seed
    params.append(f"dtype{dtype_str}")
    if seed is not None:
        params.append(f"seed{seed}")

    # Create filename
    config_str = "_".join(params)
    return f"extend_attention_inputs_{config_str}.pkl"


def create_extend_attention_inputs(
    shape: AttentionAttributes,
    dtype=torch.float16,
    seed=None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_regenerate: bool = False,
):
    """
    Create ExtendAttentionInputs with automatic save/load functionality.

    Args:
        shape: AttentionAttributes defining the configuration
        dtype: Tensor dtype to use (default: torch.float16)
        seed: Random seed for reproducible generation
        cache_dir: Directory to store cached inputs (default: ./attention_cache)
        force_regenerate: If True, ignore cached data and regenerate

    Returns:
        ExtendAttentionInputs instance
    """
    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path.cwd() / "attention_cache"
    else:
        cache_dir = Path(cache_dir)

    # Generate filename based on configuration
    dtype_str = str(dtype).split(".")[-1]  # Extract 'float16' from 'torch.float16'
    filename = _generate_config_filename(shape, dtype_str, seed)
    cache_path = cache_dir / filename

    # Try to load existing data first (unless force_regenerate is True)
    if not force_regenerate and cache_path.exists():
        try:
            print(f"Loading cached ExtendAttentionInputs from {cache_path}")
            return load_extend_attention_inputs(cache_path)
        except Exception as e:
            print(f"Warning: Failed to load cached data from {cache_path}: {e}")
            print("Regenerating data...")

    # Generate new data if no cache exists or loading failed
    print(f"Generating new ExtendAttentionInputs for config: {filename}")

    if seed:
        torch.manual_seed(seed)

    N_CTX = shape.context_len
    B = shape.num_seqs
    H_KV = shape.num_kv_heads
    H_Q = shape.num_query_heads
    D = shape.head_size
    b_seq_len_prefix = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_prefix:
        b_seq_len_prefix.fill_(shape.fixed_seq_len_prefix)
    b_seq_len_extend = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_extend:
        b_seq_len_extend.fill_(shape.fixed_seq_len_extend)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_req_idx = device_arange(B, dtype=torch.int32)
    b_start_loc = device_zeros((B,), dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = device_zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = device_zeros((B + 1,), dtype=torch.int32)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = device_zeros((b_seq_len_prefix.sum().item(),), dtype=torch.int32)

    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
        )
    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )
    v_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )

    k_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    q_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = device_empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype
        ).normal_(mean=0.1, std=0.2)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = device_zeros((B + 1,), dtype=torch.int32)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)
    logit_cap = 30.0

    b_seq_mask_len = b_seq_len_extend * b_seq_len
    # NOTE: Custom mask is of causal nature in this test. Random mask numerics
    # is not tested.
    custom_mask = device_full(
        (b_seq_mask_len.sum().item(),), fill_value=1, dtype=torch.int8
    )
    mask_offsets = device_zeros((B + 1,), dtype=torch.int32)
    mask_offsets[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
    for i in range(B):
        causal_mask = (
            torch.tril(
                device_full(
                    (b_seq_len_extend[i], b_seq_len_extend[i]),
                    fill_value=1,
                    dtype=torch.int8,
                ),
                diagonal=0,
            )
            == 1
        )
        prefix_mask = device_full(
            (b_seq_len_extend[i], b_seq_len_prefix[i]), fill_value=1, dtype=torch.int8
        )
        mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
        custom_mask[mask_offsets[i] : mask_offsets[i + 1]] = mask_flatten

    max_rpe_context_length = 10
    rpe_bias = device_zeros(max_rpe_context_length + 1, dtype=torch.float32)
    rpe_bias.copy_(device_randn(max_rpe_context_length + 1, dtype=torch.float32))
    rpe_bias[max_rpe_context_length] = 0

    output = device_zeros(
        extend_token_num, shape.num_query_heads, shape.head_size, dtype=torch.float32
    )

    # Create the inputs object
    inputs = ExtendAttentionInputs(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        output,
        max_len_extend,
        logit_cap,
    )

    # Automatically save the generated data for future use
    try:
        save_extend_attention_inputs(inputs, cache_path)
        print(f"Cached ExtendAttentionInputs to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to cache data to {cache_path}: {e}")

    return inputs


def save_extend_attention_inputs(
    inputs: ExtendAttentionInputs, filepath: Union[str, Path]
) -> None:
    """
    Save ExtendAttentionInputs instance to disk with all tensor data preserved.

    Args:
        inputs: ExtendAttentionInputs instance to save
        filepath: Path where to save the data (will create .pkl file)
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".pkl")

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data dictionary with all tensor data and metadata
    save_data = {
        "tensors": {
            "q_extend": {
                "data": inputs.q_extend.cpu(),
                "device": str(inputs.q_extend.device),
                "dtype": inputs.q_extend.dtype,
                "shape": inputs.q_extend.shape,
                "requires_grad": inputs.q_extend.requires_grad,
            },
            "k_extend": {
                "data": inputs.k_extend.cpu(),
                "device": str(inputs.k_extend.device),
                "dtype": inputs.k_extend.dtype,
                "shape": inputs.k_extend.shape,
                "requires_grad": inputs.k_extend.requires_grad,
            },
            "v_extend": {
                "data": inputs.v_extend.cpu(),
                "device": str(inputs.v_extend.device),
                "dtype": inputs.v_extend.dtype,
                "shape": inputs.v_extend.shape,
                "requires_grad": inputs.v_extend.requires_grad,
            },
            "k_buffer": {
                "data": inputs.k_buffer.cpu(),
                "device": str(inputs.k_buffer.device),
                "dtype": inputs.k_buffer.dtype,
                "shape": inputs.k_buffer.shape,
                "requires_grad": inputs.k_buffer.requires_grad,
            },
            "v_buffer": {
                "data": inputs.v_buffer.cpu(),
                "device": str(inputs.v_buffer.device),
                "dtype": inputs.v_buffer.dtype,
                "shape": inputs.v_buffer.shape,
                "requires_grad": inputs.v_buffer.requires_grad,
            },
            "qo_indptr": {
                "data": inputs.qo_indptr.cpu(),
                "device": str(inputs.qo_indptr.device),
                "dtype": inputs.qo_indptr.dtype,
                "shape": inputs.qo_indptr.shape,
                "requires_grad": inputs.qo_indptr.requires_grad,
            },
            "kv_indptr": {
                "data": inputs.kv_indptr.cpu(),
                "device": str(inputs.kv_indptr.device),
                "dtype": inputs.kv_indptr.dtype,
                "shape": inputs.kv_indptr.shape,
                "requires_grad": inputs.kv_indptr.requires_grad,
            },
            "kv_indices": {
                "data": inputs.kv_indices.cpu(),
                "device": str(inputs.kv_indices.device),
                "dtype": inputs.kv_indices.dtype,
                "shape": inputs.kv_indices.shape,
                "requires_grad": inputs.kv_indices.requires_grad,
            },
            "output": {
                "data": inputs.output.cpu(),
                "device": str(inputs.output.device),
                "dtype": inputs.output.dtype,
                "shape": inputs.output.shape,
                "requires_grad": inputs.output.requires_grad,
            },
        },
        "scalars": {
            "max_len_extend": inputs.max_len_extend,
            "logit_cap": inputs.logit_cap,
        },
        "metadata": {
            "save_format_version": "1.0",
            "torch_version": str(torch.__version__),
        },
    }

    # Save using torch.save for better tensor serialization
    torch.save(save_data, filepath)
    print(f"ExtendAttentionInputs saved to {filepath}")


def load_extend_attention_inputs(
    filepath: Union[str, Path], device: Optional[str] = None
) -> ExtendAttentionInputs:
    """
    Load ExtendAttentionInputs instance from disk with exact tensor data restored.

    Args:
        filepath: Path to the saved data file
        device: Target device for loaded tensors (if None, uses original device)

    Returns:
        ExtendAttentionInputs instance with all tensor data restored
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".pkl")

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load the data with weights_only=False for backward compatibility
    # This is safe since we control the file format and contents
    try:
        save_data = torch.load(filepath, map_location="cpu", weights_only=False)
    except Exception as e:
        # Fallback for older PyTorch versions that don't support weights_only
        save_data = torch.load(filepath, map_location="cpu")

    # Extract tensors and restore their properties
    tensors = {}
    for tensor_name, tensor_info in save_data["tensors"].items():
        # Restore tensor data
        tensor_data = tensor_info["data"]

        # Determine target device
        target_device = device if device is not None else tensor_info["device"]

        # Move to target device and restore properties
        restored_tensor = tensor_data.to(
            device=target_device, dtype=tensor_info["dtype"]
        )

        if tensor_info["requires_grad"]:
            restored_tensor.requires_grad_(True)

        tensors[tensor_name] = restored_tensor

    # Extract scalars
    scalars = save_data["scalars"]

    # Create and return the ExtendAttentionInputs instance
    return ExtendAttentionInputs(
        q_extend=tensors["q_extend"],
        k_extend=tensors["k_extend"],
        v_extend=tensors["v_extend"],
        k_buffer=tensors["k_buffer"],
        v_buffer=tensors["v_buffer"],
        qo_indptr=tensors["qo_indptr"],
        kv_indptr=tensors["kv_indptr"],
        kv_indices=tensors["kv_indices"],
        output=tensors["output"],
        max_len_extend=scalars["max_len_extend"],
        logit_cap=scalars["logit_cap"],
    )


def clear_extend_attention_cache(cache_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Clear all cached ExtendAttentionInputs files.

    Args:
        cache_dir: Cache directory to clear (default: ./attention_cache)
    """
    import shutil

    if cache_dir is None:
        cache_dir = Path.cwd() / "attention_cache"
    else:
        cache_dir = Path(cache_dir)

    if cache_dir.exists():
        cache_files = list(cache_dir.glob("extend_attention_inputs_*.pkl"))
        if cache_files:
            for cache_file in cache_files:
                cache_file.unlink()
            print(f"Cleared {len(cache_files)} cache files from {cache_dir}")
        else:
            print(f"No cache files found in {cache_dir}")
    else:
        print(f"Cache directory {cache_dir} does not exist")
