from dataclasses import dataclass
from typing import override

# Import from new config module
from kernel_bench.config.types.gemm import GemmConfig

# Re-export for backwards compatibility
__all__ = ["GemmConfig"]
